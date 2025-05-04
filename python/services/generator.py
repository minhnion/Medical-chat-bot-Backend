
import os
# Thay thế import OpenAI bằng import thư viện Google
import google.generativeai as genai
from dotenv import load_dotenv
import logging
import time # Thêm để xử lý rate limit cơ bản

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Tải biến môi trường từ file .env ở thư mục gốc
# Đảm bảo đường dẫn này chính xác trỏ đến file .env gốc của dự án
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'backend/.env')

if not os.path.exists(dotenv_path):
    logging.error(f".env file not found at expected path: {dotenv_path}")

load_dotenv(dotenv_path=dotenv_path, verbose=True) # verbose=True để xem log load dotenv

# --- Cấu hình ---
# Lấy Google API Key từ .env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logging.error("GOOGLE_API_KEY không được đặt trong file .env! GeneratorService sẽ không thể hoạt động với Google Gemini.")
    # Không raise lỗi ngay để cho phép import class, nhưng các hàm sẽ không hoạt động

# Sử dụng tên model Gemini ổn định (thay vì 'gemini-pro')
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "models/gemini-1.5-flash-latest")
# --- Kết thúc Cấu hình ---

class GeneratorService:
    """
    Dịch vụ sinh văn bản sử dụng Google Gemini API.
    """
    def __init__(self, api_key=GOOGLE_API_KEY, model_name=GEMINI_MODEL_NAME):
        """
        Khởi tạo service, cấu hình API key và chuẩn bị model Gemini.
        """
        logging.info(f"Khởi tạo GeneratorService với model Google Gemini: {model_name}")
        self.model = None
        self.model_name = model_name
        self.api_key_configured = False # Thêm cờ trạng thái cấu hình

        if not api_key:
            logging.error("Không có Google API Key được cung cấp. Không thể cấu hình Gemini.")
        else:
            try:
                # Cấu hình API key cho thư viện google-generativeai
                genai.configure(api_key=api_key)
                self.api_key_configured = True # Đánh dấu là key đã được cấu hình

                # Khởi tạo model Gemini
                # Việc này cũng giúp kiểm tra sớm xem model có hợp lệ không
                logging.info(f"Đang khởi tạo model Gemini: {self.model_name}...")
                self.model = genai.GenerativeModel(self.model_name)
                # Có thể thêm một lần gọi nhỏ để kiểm tra kết nối/key ngay lập tức
                # Ví dụ: list_models (có thể tốn request nhỏ)
                # genai.list_models()
                logging.info(f"Thư viện Google Generative AI đã được cấu hình và model '{self.model_name}' đã được khởi tạo.")

            except Exception as e:
                # Bắt các lỗi có thể xảy ra khi khởi tạo (vd: key sai định dạng, tên model không tồn tại)
                logging.error(f"Lỗi khi cấu hình Google Generative AI hoặc khởi tạo model '{self.model_name}': {e}", exc_info=True) # Thêm exc_info để xem traceback
                self.model = None # Đảm bảo model là None nếu có lỗi
                self.api_key_configured = False # Đánh dấu cấu hình thất bại

    def _create_prompt(self, query: str, context: list) -> str:
        """
        Tạo prompt chi tiết cho mô hình Gemini từ câu hỏi và ngữ cảnh.
        """
        if not context:
            logging.warning("Không có context được cung cấp cho prompt.")
            context_str = "Không có thông tin bổ sung nào được cung cấp."
        else:
            # Nối các đoạn context lại, có thể đánh số cho rõ ràng
            context_items = [f"Thông tin {i+1}: {ctx}" for i, ctx in enumerate(context)]
            context_str = "\n\n".join(context_items)

        # Mẫu prompt được tinh chỉnh
        prompt = f"""**Thông tin tham khảo:**

{context_str}

---
**Yêu cầu:**

Dựa vào **Thông tin tham khảo** ở trên, hãy trả lời câu hỏi sau một cách chính xác và hữu ích.
*   Nếu thông tin có trong tham khảo, hãy tóm tắt và trình bày lại.
*   Nếu thông tin không có trong tham khảo, hãy trả lời dựa trên kiến thức y tế chung của bạn nhưng **phải** nói rõ rằng thông tin này không có trong tài liệu tham khảo được cung cấp.
*   Luôn trả lời một cách cẩn trọng, đặc biệt với thông tin y tế. Tránh đưa ra chẩn đoán trực tiếp hoặc lời khuyên thay thế chuyên gia y tế.
*   Không bịa đặt thông tin.

**Câu hỏi:** {query}

**Trả lời:**
"""
        # Dùng logging.debug cho prompt để tránh log quá nhiều khi chạy bình thường
        logging.debug(f"Prompt được tạo (độ dài: {len(prompt)} chars)")
        return prompt

    def generate_response(self, query: str, context: list, max_retries=2, initial_delay=1) -> str:
        """
        Gửi yêu cầu đến Google Gemini API và nhận phản hồi, có xử lý retry.

        Args:
            query (str): Câu hỏi của người dùng.
            context (list): Danh sách các chuỗi ngữ cảnh từ retriever.
            max_retries (int): Số lần thử lại tối đa nếu gặp lỗi rate limit.
            initial_delay (int): Thời gian chờ ban đầu (giây) trước khi thử lại.

        Returns:
            str: Câu trả lời do Gemini tạo ra hoặc thông báo lỗi.
        """
        # Kiểm tra trạng thái cấu hình và khởi tạo model trước khi thực hiện
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

                # Cấu hình generation (tùy chọn)
                generation_config = genai.types.GenerationConfig(
                    max_output_tokens=350, # Tăng nhẹ giới hạn output
                    temperature=0.7
                )

                # Cấu hình an toàn (điều chỉnh nếu cần)
                safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                ]

                # Gọi API
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )

                # --- Xử lý phản hồi ---
                # Kiểm tra an toàn nhất là xem candidate có tồn tại không
                if not response.candidates:
                     try:
                         reason = response.prompt_feedback.block_reason
                         error_msg = f"Lỗi: Yêu cầu bị chặn bởi bộ lọc an toàn của Google (Lý do: {reason})."
                         logging.error(error_msg)
                         return error_msg
                     except Exception:
                         logging.error("Yêu cầu không có candidates trả về, có thể do bị chặn hoặc lỗi không xác định.")
                         return "Lỗi: Không nhận được phản hồi hợp lệ từ Google (có thể do bộ lọc an toàn)."

                # Kiểm tra xem candidate có content parts không
                if not response.candidates[0].content.parts:
                    logging.error("Phản hồi có candidate nhưng không có content parts.")
                    return "Lỗi: Phản hồi từ Google không chứa nội dung văn bản."

                # Lấy text từ part đầu tiên
                answer = response.candidates[0].content.parts[0].text.strip()
                logging.info("Nhận được phản hồi từ Google Gemini API.")
                return answer # Trả về kết quả thành công

            except Exception as e:
                error_str = str(e).lower()
                # Kiểm tra lỗi Rate Limit (429) hoặc Resource Exhausted
                if "429" in error_str or "resource has been exhausted" in error_str or "rate limit" in error_str:
                    if retries < max_retries:
                        logging.warning(f"Gặp lỗi Rate Limit Google Gemini (429). Đang chờ {delay} giây để thử lại...")
                        time.sleep(delay)
                        retries += 1
                        delay *= 2 # Exponential backoff
                        continue # Quay lại đầu vòng lặp để thử lại
                    else:
                        logging.error(f"Gặp lỗi Rate Limit Google Gemini (429) và đã hết số lần thử lại.")
                        # Ném lỗi hoặc trả về thông báo lỗi
                        return "Lỗi: Đã đạt giới hạn yêu cầu miễn phí của Google Gemini. Vui lòng thử lại sau." # Trả về thông báo lỗi

                # Kiểm tra lỗi API Key không hợp lệ
                elif "api key not valid" in error_str or "permission denied" in error_str or "authentication" in error_str:
                     logging.error(f"Lỗi xác thực API Key Google: {e}")
                     return "Lỗi: API Key của Google không hợp lệ hoặc không có quyền truy cập model này." # Trả về thông báo lỗi

                # Kiểm tra lỗi Model không tìm thấy
                elif "404" in error_str and f"models/{self.model_name}" in error_str:
                     logging.error(f"Lỗi không tìm thấy model Google Gemini '{self.model_name}': {e}")
                     return f"Lỗi: Không tìm thấy model '{self.model_name}' trên Google AI. Vui lòng kiểm tra lại tên model." # Trả về thông báo lỗi

                # Các lỗi khác
                else:
                    logging.error(f"Lỗi không xác định khi gọi Google Gemini API: {e}", exc_info=True)
                    # Trả về thông báo lỗi chung
                    return f"Xin lỗi, đã xảy ra lỗi khi tạo câu trả lời qua Google Gemini: {type(e).__name__}"

        # Nếu vòng lặp kết thúc mà không thành công (chỉ khi retry hết hoặc lỗi không retry được)
        logging.error("Không thể nhận phản hồi từ Google Gemini sau các lần thử.")
        return "Lỗi: Không thể nhận phản hồi từ Google Gemini sau các lần thử."


# --- Chạy trực tiếp để kiểm tra nhanh ---
if __name__ == "__main__":
    logging.info("Bắt đầu kiểm tra GeneratorService với Google Gemini...")
    generator = None
    # Chỉ chạy test nếu có API Key
    if not GOOGLE_API_KEY:
         logging.error("Thiếu GOOGLE_API_KEY trong .env. Bỏ qua kiểm tra GeneratorService.")
    else:
        try:
            # Khởi tạo service
            generator = GeneratorService() # Sẽ dùng key và model từ config

            # Chỉ chạy test nếu model được khởi tạo thành công
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
                time.sleep(2) # Tăng thời gian chờ

                print(f"\n--- Thử tạo câu trả lời cho query (không context) qua Google Gemini: '{test_query}' ---")
                response2 = generator.generate_response(test_query, empty_context)
                print(f"\nCâu trả lời (không context):\n{response2}")
            else:
                 logging.error("Không thể chạy kiểm tra vì GeneratorService hoặc model Gemini chưa được khởi tạo thành công (kiểm tra lỗi ở trên, API key và tên model).")
                 print("\nKhông thể chạy kiểm tra vì GeneratorService hoặc model Gemini chưa được khởi tạo thành công.")

        except Exception as e:
            # Bắt lỗi chung trong quá trình kiểm tra
            logging.error(f"Lỗi không mong muốn xảy ra trong quá trình kiểm tra: {e}", exc_info=True)

    logging.info("Kết thúc kiểm tra GeneratorService với Google Gemini.")