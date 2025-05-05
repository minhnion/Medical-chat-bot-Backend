import axios from 'axios';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

// Cấu hình để đọc .env từ thư mục gốc (quan trọng nếu bạn dùng ES Modules)
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
// Đi lên 2 cấp từ services -> src -> backend rồi vào file .env gốc
dotenv.config({ path: path.resolve(__dirname, '..', '..', '.env') });

const PYTHON_API_BASE_URL = process.env.PYTHON_API_URL;

if (!PYTHON_API_BASE_URL) {
  console.error("Lỗi: Biến môi trường PYTHON_API_URL chưa được đặt!");
  // Có thể throw lỗi hoặc dùng giá trị mặc định (không khuyến khích)
  // throw new Error("PYTHON_API_URL is not set in the environment variables.");
}

/**
 * Gửi query đến Python RAG API và nhận câu trả lời.
 * @param {string} query - Câu hỏi của người dùng.
 * @returns {Promise<string>} - Promise chứa câu trả lời từ bot.
 * @throws {Error} - Ném lỗi nếu có vấn đề khi gọi API hoặc API trả về lỗi.
 */
const getRagResponse = async (query) => {
  if (!PYTHON_API_BASE_URL) {
     throw new Error("Python API URL is not configured.");
  }
  console.log(`[PythonService] Gửi query tới ${PYTHON_API_BASE_URL}/chat: "${query}"`);
  try {
    const response = await axios.post(`${PYTHON_API_BASE_URL}/chat`, {
      query: query, // Gửi query trong body dạng JSON
    }, {
      headers: {
        'Content-Type': 'application/json',
      },
       timeout: 60000 // Thêm timeout (vd: 60 giây) để tránh chờ đợi quá lâu
    });

    // Kiểm tra xem response có dữ liệu và trường 'answer' không
    if (response.data && response.data.answer) {
      console.log("[PythonService] Nhận được answer:", response.data.answer.substring(0, 100) + "..."); // Log phần đầu
      return response.data.answer;
    } else if (response.data && response.data.error) {
       // Nếu Python API trả về lỗi JSON dạng {"error": "..."}
       console.error("[PythonService] Python API trả về lỗi:", response.data.error);
       throw new Error(`Lỗi từ Python Service: ${response.data.error}`);
    } else {
      // Trường hợp trả về không mong muốn
      console.error("[PythonService] Phản hồi không hợp lệ từ Python API:", response.data);
      throw new Error("Phản hồi không hợp lệ từ dịch vụ Python.");
    }
  } catch (error) {
    console.error("[PythonService] Lỗi khi gọi Python API:", error.message);
    // Xử lý các loại lỗi khác nhau từ axios
    if (error.response) {
      // Server Python đã trả về lỗi (status code không phải 2xx)
      console.error(`[PythonService] Lỗi từ server Python (${error.response.status}):`, error.response.data);
      const pythonErrorMsg = error.response.data?.error || `Lỗi ${error.response.status} từ server Python`;
      throw new Error(`Lỗi từ Python Service: ${pythonErrorMsg}`);
    } else if (error.request) {
      // Yêu cầu đã được gửi nhưng không nhận được phản hồi (vd: server Python không chạy, timeout)
      console.error("[PythonService] Không nhận được phản hồi từ server Python.");
      throw new Error("Không thể kết nối đến dịch vụ Python.");
    } else {
      // Lỗi xảy ra khi thiết lập yêu cầu
      console.error("[PythonService] Lỗi thiết lập yêu cầu:", error.message);
      throw new Error("Lỗi khi chuẩn bị yêu cầu đến dịch vụ Python.");
    }
  }
};

export const pythonService = {
  getRagResponse,
};

