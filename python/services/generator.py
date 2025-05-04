import os
import google.generativeai as genai # Import thư viện Google
from dotenv import load_dotenv
import logging
import time # Thêm để xử lý rate limit cơ bản

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Tải biến môi trường
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'backend/.env')
load_dotenv(dotenv_path=dotenv_path)

# --- Cấu hình ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logging.warning("GOOGLE_API_KEY không được đặt trong file .env. GeneratorService sẽ không hoạt động với Google Gemini.")

# Chọn model Gemini (ví dụ: 'gemini-pro')
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")
# --- Kết thúc Cấu hình ---

class GeneratorService:
    def __init__(self, api_key=GOOGLE_API_KEY, model_name=GEMINI_MODEL):
        """Khởi tạo service với API key và model của Google Gemini."""
        logging.info(f"Khởi tạo GeneratorService với model Google Gemini: {model_name}")
        self.model = None
        if not api_key:
            logging.error("Không có Google API Key. Không thể cấu hình Gemini.")
        else:
            try:
                # Cấu hình API key cho thư viện
                genai.configure(api_key=api_key)
                # Khởi tạo model (có thể kiểm tra model tồn tại nếu cần)
                # Lưu ý: Cách khởi tạo này có thể thay đổi tuỳ phiên bản thư viện
                # Cách tiếp cận đơn giản là chỉ cần configure key, model sẽ được gọi khi generate
                self.model_name = model_name
                # Tạo instance model để dùng sau (tùy chọn, có thể tạo lúc gọi)
                self.model = genai.GenerativeModel(self.model_name)
                logging.info("Thư viện Google Generative AI đã được cấu hình.")
            except Exception as e:
                logging.error(f"Lỗi khi cấu hình Google Generative AI: {e}")
                self.model = None # Đảm bảo model là None nếu có lỗi

    def _create_prompt(self, query: str, context: list) -> str:
        # Hàm này giữ nguyên, cấu trúc prompt có thể dùng chung
        if not context:
            logging.warning("Không có context được cung cấp, prompt sẽ chỉ dựa trên query.")
            context_str = "Không tìm thấy thông tin liên quan trong dữ liệu."
        else:
            context_str = "\n\n".join(context)

        # Có thể điều chỉnh prompt một chút nếu cần cho Gemini
        prompt = f"""Dựa vào thông tin dưới đây:
--- Context ---
{context_str}
--- Hết Context ---

Hãy trả lời câu hỏi sau một cách chính xác và ngắn gọn nhất có thể, tập trung vào thông tin trong context nếu có liên quan. Nếu context không đủ thông tin, hãy trả lời dựa trên kiến thức chung của bạn về y tế nhưng nói rõ là thông tin không có trong context. Không tự ý bịa đặt thông tin không có căn cứ.

Câu hỏi: {query}

Trả lời:"""
        logging.info(f"Prompt được tạo (độ dài: {len(prompt)} chars)")
        return prompt

    def generate_response(self, query: str, context: list, max_retries=2, initial_delay=1) -> str:
        """
        Gửi prompt đến Google Gemini API và nhận phản hồi.
        Thêm retry cơ bản để xử lý Rate Limit.
        """
        if not self.model: # Kiểm tra xem model đã được khởi tạo chưa
            error_message = "Model Google Gemini chưa được khởi tạo (thiếu API key hoặc lỗi cấu hình)."
            logging.error(error_message)
            return error_message

        prompt = self._create_prompt(query, context)
        retries = 0
        delay = initial_delay

        while retries <= max_retries:
            try:
                logging.info(f"Đang gửi yêu cầu tới model Google Gemini: {self.model_name} (Lần thử {retries + 1})...")
                # Sử dụng generate_content thay vì chat.completions.create
                response = self.model.generate_content(prompt)

                # Kiểm tra xem có bị block không (do safety settings)
                if not response.parts:
                     # Thử truy cập prompt_feedback để xem lý do bị block
                     try:
                         reason = response.prompt_feedback.block_reason
                         logging.error(f"Yêu cầu bị chặn bởi Google Safety Settings. Lý do: {reason}")
                         return f"Lỗi: Yêu cầu bị chặn bởi bộ lọc an toàn của Google (Lý do: {reason}). Hãy thử thay đổi câu hỏi hoặc nội dung."
                     except Exception:
                         logging.error("Yêu cầu bị chặn bởi Google Safety Settings (không rõ lý do cụ thể).")
                         return "Lỗi: Yêu cầu bị chặn bởi bộ lọc an toàn của Google."


                # Truy cập kết quả text
                answer = response.text.strip()
                logging.info("Nhận được phản hồi từ Google Gemini API.")
                return answer

            except Exception as e:
                # Kiểm tra lỗi Rate Limit (thường là ResourceExhaustedError hoặc tương tự)
                # Mã lỗi 429 trong string representation của lỗi là dấu hiệu tốt
                if "429" in str(e) or "Resource has been exhausted" in str(e) or "rate limit" in str(e).lower():
                    if retries < max_retries:
                        logging.warning(f"Gặp lỗi Rate Limit (429). Đang chờ {delay} giây để thử lại...")
                        time.sleep(delay)
                        retries += 1
                        delay *= 2 # Tăng thời gian chờ (exponential backoff)
                    else:
                        logging.error(f"Gặp lỗi Rate Limit (429) và đã hết số lần thử lại.")
                        return "Lỗi: Đã đạt giới hạn yêu cầu miễn phí của Google Gemini (60 RPM). Vui lòng thử lại sau."
                elif "API key not valid" in str(e):
                     logging.error(f"Lỗi xác thực API Key Google: {e}")
                     return "Lỗi: API Key của Google không hợp lệ. Vui lòng kiểm tra lại."
                else:
                    # Các lỗi khác
                    logging.error(f"Lỗi khi gọi Google Gemini API: {e}", exc_info=True) # Thêm exc_info để xem traceback
                    return f"Xin lỗi, đã xảy ra lỗi khi tạo câu trả lời qua Google Gemini: {type(e).__name__}"
        # Nếu vòng lặp kết thúc mà không thành công (chỉ xảy ra nếu max_retries=0 và gặp lỗi ngay)
        return "Lỗi: Không thể nhận phản hồi từ Google Gemini sau các lần thử."


# --- Chạy trực tiếp để kiểm tra nhanh ---
if __name__ == "__main__":
    logging.info("Bắt đầu kiểm tra GeneratorService với Google Gemini...")
    generator = None
    if not GOOGLE_API_KEY:
         logging.error("Thiếu GOOGLE_API_KEY trong .env. Không thể chạy kiểm tra.")
    else:
        try:
            generator = GeneratorService()

            test_query = "Triệu chứng đau đầu và sốt nhẹ là bệnh gì?"
            test_context = [
                "Bệnh nhân A có triệu chứng đau đầu kéo dài, kèm theo sốt nhẹ vào buổi chiều. Chẩn đoán sơ bộ là cảm cúm thông thường.",
                "Sốt nhẹ và đau đầu cũng có thể là dấu hiệu ban đầu của sốt xuất huyết, cần theo dõi thêm các dấu hiệu xuất huyết.",
                "Một số trường hợp viêm xoang cũng gây đau đầu và có thể kèm sốt nhẹ."
            ]
            empty_context = []

            if generator.model: # Kiểm tra model đã khởi tạo chưa
                 print(f"\n--- Thử tạo câu trả lời cho query (có context) qua Google Gemini: '{test_query}' ---")
                 response1 = generator.generate_response(test_query, test_context)
                 print(f"\nCâu trả lời (có context):\n{response1}")

                 # Thêm delay nhỏ giữa các lần gọi để tránh rate limit ngay lập tức khi test
                 print("\n(Đợi 1 giây trước khi gọi lần 2...)")
                 time.sleep(1)

                 print(f"\n--- Thử tạo câu trả lời cho query (không context) qua Google Gemini: '{test_query}' ---")
                 response2 = generator.generate_response(test_query, empty_context)
                 print(f"\nCâu trả lời (không context):\n{response2}")
            else:
                 print("\nKhông thể chạy kiểm tra vì model Google Gemini chưa được khởi tạo.")

        except Exception as e:
            logging.error(f"Lỗi không mong muốn trong quá trình kiểm tra Generator với Google Gemini: {e}", exc_info=True)

    logging.info("Kết thúc kiểm tra GeneratorService với Google Gemini.")