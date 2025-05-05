import axios from 'axios';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
dotenv.config({ path: path.resolve(__dirname, '..', '..', '.env') });

const PYTHON_API_BASE_URL = process.env.PYTHON_API_URL;

if (!PYTHON_API_BASE_URL) {
  console.error("Lỗi: Biến môi trường PYTHON_API_URL chưa được đặt!");

}

const getRagResponse = async (query) => {
  if (!PYTHON_API_BASE_URL) {
     throw new Error("Python API URL is not configured.");
  }
  console.log(`[PythonService] Gửi query tới ${PYTHON_API_BASE_URL}/chat: "${query}"`);
  try {
    const response = await axios.post(`${PYTHON_API_BASE_URL}/chat`, {
      query: query, 
    }, {
      headers: {
        'Content-Type': 'application/json',
      },
       timeout: 60000 
    });

    if (response.data && response.data.answer) {
      console.log("[PythonService] Nhận được answer:", response.data.answer.substring(0, 100) + "..."); // Log phần đầu
      return response.data.answer;
    } else if (response.data && response.data.error) {
       console.error("[PythonService] Python API trả về lỗi:", response.data.error);
       throw new Error(`Lỗi từ Python Service: ${response.data.error}`);
    } else {
      console.error("[PythonService] Phản hồi không hợp lệ từ Python API:", response.data);
      throw new Error("Phản hồi không hợp lệ từ dịch vụ Python.");
    }
  } catch (error) {
    console.error("[PythonService] Lỗi khi gọi Python API:", error.message);
    if (error.response) {
      console.error(`[PythonService] Lỗi từ server Python (${error.response.status}):`, error.response.data);
      const pythonErrorMsg = error.response.data?.error || `Lỗi ${error.response.status} từ server Python`;
      throw new Error(`Lỗi từ Python Service: ${pythonErrorMsg}`);
    } else if (error.request) {
      console.error("[PythonService] Không nhận được phản hồi từ server Python.");
      throw new Error("Không thể kết nối đến dịch vụ Python.");
    } else {
      console.error("[PythonService] Lỗi thiết lập yêu cầu:", error.message);
      throw new Error("Lỗi khi chuẩn bị yêu cầu đến dịch vụ Python.");
    }
  }
};

export const pythonService = {
  getRagResponse,
};

