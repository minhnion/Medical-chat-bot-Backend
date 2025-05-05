
import os
import google.generativeai as genai
from dotenv import load_dotenv
import logging
import time 

# loggin config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# .env file path
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'backend/.env')

if not os.path.exists(dotenv_path):
    logging.error(f".env file not found at expected path: {dotenv_path}")

load_dotenv(dotenv_path=dotenv_path, verbose=True) # verbose=True để xem log load dotenv

# config
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logging.error("GOOGLE_API_KEY không được đặt trong file .env! GeneratorService sẽ không thể hoạt động với Google Gemini.")

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "models/gemini-1.5-flash-latest")

class GeneratorService:

    def __init__(self, api_key=GOOGLE_API_KEY, model_name=GEMINI_MODEL_NAME):

        logging.info(f"Khởi tạo GeneratorService với model Google Gemini: {model_name}")
        self.model = None
        self.model_name = model_name
        self.api_key_configured = False 

        if not api_key:
            logging.error("Không có Google API Key được cung cấp. Không thể cấu hình Gemini.")
        else:
            try:
                genai.configure(api_key=api_key)
                self.api_key_configured = True 

                logging.info(f"Đang khởi tạo model Gemini: {self.model_name}...")
                self.model = genai.GenerativeModel(self.model_name)
                logging.info(f"Thư viện Google Generative AI đã được cấu hình và model '{self.model_name}' đã được khởi tạo.")

            except Exception as e:
                logging.error(f"Lỗi khi cấu hình Google Generative AI hoặc khởi tạo model '{self.model_name}': {e}", exc_info=True) # Thêm exc_info để xem traceback
                self.model = None
                self.api_key_configured = False

    def _create_prompt(self, query: str, context: list) -> str:
        if not context:
            logging.warning("Không có context được cung cấp cho prompt.")
            context_str = "Không có thông tin bổ sung nào được cung cấp."
        else:
            context_items = [f"Thông tin {i+1}: {ctx}" for i, ctx in enumerate(context)]
            context_str = "\n\n".join(context_items)

        prompt = f"""**Thông tin tham khảo:**

{context_str}


**Câu hỏi:** {query}

"""
        logging.debug(f"Prompt được tạo (độ dài: {len(prompt)} chars)")
        return prompt

    def generate_response(self, query: str, context: list, max_retries=2, initial_delay=1) -> str:
        if not self.api_key_configured:
             logging.error("Không thể tạo phản hồi: API Key của Google chưa được cấu hình.")
             return "Lỗi: API Key của Google chưa được cấu hình."
        if not self.model:
            logging.error(f"Không thể tạo phản hồi: Model Gemini '{self.model_name}' chưa được khởi tạo thành công.")
            return f"Lỗi: Model Gemini '{self.model_name}' chưa được khởi tạo thành công."

        prompt = self._create_prompt(query, context)
        retries = 0
        delay = initial_delay

        while retries <= max_retries:
            try:
                logging.info(f"Đang gửi yêu cầu tới model Google Gemini: {self.model_name} (Lần thử {retries + 1})...")

                generation_config = genai.types.GenerationConfig(
                    max_output_tokens=500,
                    temperature=0.7
                )

                safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                ]

                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )

                if not response.candidates:
                     try:
                         reason = response.prompt_feedback.block_reason
                         error_msg = f"Lỗi: Yêu cầu bị chặn bởi bộ lọc an toàn của Google (Lý do: {reason})."
                         logging.error(error_msg)
                         return error_msg
                     except Exception:
                         logging.error("Yêu cầu không có candidates trả về, có thể do bị chặn hoặc lỗi không xác định.")
                         return "Lỗi: Không nhận được phản hồi hợp lệ từ Google (có thể do bộ lọc an toàn)."

                if not response.candidates[0].content.parts:
                    logging.error("Phản hồi có candidate nhưng không có content parts.")
                    return "Lỗi: Phản hồi từ Google không chứa nội dung văn bản."

                answer = response.candidates[0].content.parts[0].text.strip()
                logging.info("Nhận được phản hồi từ Google Gemini API.")
                return answer 

            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "resource has been exhausted" in error_str or "rate limit" in error_str:
                    if retries < max_retries:
                        logging.warning(f"Gặp lỗi Rate Limit Google Gemini (429). Đang chờ {delay} giây để thử lại...")
                        time.sleep(delay)
                        retries += 1
                        delay *= 2 
                        continue 
                    else:
                        logging.error(f"Gặp lỗi Rate Limit Google Gemini (429) và đã hết số lần thử lại.")
                        return "Lỗi: Đã đạt giới hạn yêu cầu miễn phí của Google Gemini. Vui lòng thử lại sau." # Trả về thông báo lỗi

                elif "api key not valid" in error_str or "permission denied" in error_str or "authentication" in error_str:
                     logging.error(f"Lỗi xác thực API Key Google: {e}")
                     return "Lỗi: API Key của Google không hợp lệ hoặc không có quyền truy cập model này." # Trả về thông báo lỗi

                elif "404" in error_str and f"models/{self.model_name}" in error_str:
                     logging.error(f"Lỗi không tìm thấy model Google Gemini '{self.model_name}': {e}")
                     return f"Lỗi: Không tìm thấy model '{self.model_name}' trên Google AI. Vui lòng kiểm tra lại tên model." # Trả về thông báo lỗi

                else:
                    logging.error(f"Lỗi không xác định khi gọi Google Gemini API: {e}", exc_info=True)
                    return f"Xin lỗi, đã xảy ra lỗi khi tạo câu trả lời qua Google Gemini: {type(e).__name__}"

        logging.error("Không thể nhận phản hồi từ Google Gemini sau các lần thử.")
        return "Lỗi: Không thể nhận phản hồi từ Google Gemini sau các lần thử."


if __name__ == "__main__":
    logging.info("Bắt đầu kiểm tra GeneratorService với Google Gemini...")
    generator = None
    if not GOOGLE_API_KEY:
         logging.error("Thiếu GOOGLE_API_KEY trong .env. Bỏ qua kiểm tra GeneratorService.")
    else:
        try:
            generator = GeneratorService() 

            if generator and generator.model:
                logging.info("GeneratorService và model đã khởi tạo thành công. Bắt đầu gửi yêu cầu test...")
                test_query = "Triệu chứng đau đầu và sốt nhẹ là bệnh gì?"
                test_context = [
                    "Bệnh nhân A có triệu chứng đau đầu kéo dài, kèm theo sốt nhẹ vào buổi chiều. Chẩn đoán sơ bộ là cảm cúm thông thường.",
                    "Sốt nhẹ và đau đầu cũng có thể là dấu hiệu ban đầu của sốt xuất huyết, cần theo dõi thêm các dấu hiệu xuất huyết.",
                    "Một số trường hợp viêm xoang cũng gây đau đầu và có thể kèm sốt nhẹ."
                ]
                empty_context = []

                print(f"\n--- Thử tạo câu trả lời cho query (có context) qua Google Gemini: '{test_query}' ---")
                response1 = generator.generate_response(test_query, test_context)
                print(f"\nCâu trả lời (có context):\n{response1}")

                print("\n(Đợi 1-2 giây trước khi gọi lần 2 để giảm khả năng bị rate limit...)")
                time.sleep(2) 

                print(f"\n--- Thử tạo câu trả lời cho query (không context) qua Google Gemini: '{test_query}' ---")
                response2 = generator.generate_response(test_query, empty_context)
                print(f"\nCâu trả lời (không context):\n{response2}")
            else:
                 logging.error("Không thể chạy kiểm tra vì GeneratorService hoặc model Gemini chưa được khởi tạo thành công (kiểm tra lỗi ở trên, API key và tên model).")
                 print("\nKhông thể chạy kiểm tra vì GeneratorService hoặc model Gemini chưa được khởi tạo thành công.")

        except Exception as e:
            logging.error(f"Lỗi không mong muốn xảy ra trong quá trình kiểm tra: {e}", exc_info=True)

    logging.info("Kết thúc kiểm tra GeneratorService với Google Gemini.")