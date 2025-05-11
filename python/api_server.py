import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS 
from services.retriever import RetrieverService
from services.generator import GeneratorService 
from deep_translator import GoogleTranslator
from langdetect import detect


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)
logging.info("Ứng dụng Flask đang được khởi tạo...")

try:
    logging.info("Đang khởi tạo RetrieverService...")
    retriever = RetrieverService()
    logging.info("RetrieverService đã sẵn sàng.")
except Exception as e:
    logging.error(f"LỖI NGHIÊM TRỌNG: Không thể khởi tạo RetrieverService: {e}", exc_info=True)
    retriever = None 

try:
    logging.info("Đang khởi tạo GeneratorService...")
    generator = GeneratorService()
    if generator and not getattr(generator, 'client', None) and not getattr(generator, 'model', None):
         logging.warning("GeneratorService được tạo nhưng client/model bên trong có thể chưa sẵn sàng (thiếu API key?).")
    elif generator:
         logging.info("GeneratorService đã sẵn sàng.")
    else:
         pass

except Exception as e:
    logging.error(f"LỖI NGHIÊM TRỌNG: Không thể khởi tạo GeneratorService: {e}", exc_info=True)
    generator = None 


# API endpoint
@app.route('/chat', methods=['POST'])
def handle_chat():
    logging.info("Nhận được yêu cầu tới /chat")
    if not retriever:
        logging.error("RetrieverService không khả dụng.")
        return jsonify({"error": "Dịch vụ tìm kiếm không khả dụng."}), 503 

    if not generator:
        logging.error("GeneratorService không khả dụng.")
        return jsonify({"error": "Dịch vụ sinh câu trả lời không khả dụng."}), 503

    data = request.get_json()
    if not data or 'query' not in data:
        logging.warning("Yêu cầu không hợp lệ: Thiếu 'query' trong JSON body.")
        return jsonify({"error": "Thiếu trường 'query' trong yêu cầu JSON."}), 400

    query = data['query']
    logging.info(f"Query nhận được: '{query}'")

    # Phát hiện ngôn ngữ của câu hỏi
    try:
        detected_language = detect(query)
        logging.info(f"Ngôn ngữ phát hiện: {detected_language}")
    except Exception as e:
        logging.error(f"Lỗi khi phát hiện ngôn ngữ: {e}")
        return jsonify({"error": "Không thể phát hiện ngôn ngữ của câu hỏi."}), 500

    # Dịch câu hỏi sang tiếng Anh nếu cần
    if detected_language == 'vi':
        logging.info("Phát hiện ngôn ngữ đầu vào là tiếng Việt. Đang dịch sang tiếng Anh...")
        query = GoogleTranslator(source='vi', target='en').translate(query)
        logging.info(f"Câu hỏi sau khi dịch sang tiếng Anh: '{query}'")

    try:
        logging.info("Bắt đầu quá trình Retrieve...")
        retrieved_results = retriever.retrieve(query, top_k=3, fetch_context=True)

        contexts = [result.get('context', '') for result in retrieved_results if result.get('context')]
        
        logging.info(f"Số context tìm thấy để gửi cho Generator: {len(contexts)}")
        if contexts:
            for i, ctx in enumerate(contexts):
                logging.info(f"Context {i+1} thực tế: {ctx[:300]}...")
        else:
            logging.warning("Không có context nào được tìm thấy hoặc trích xuất được.")
    
        if not contexts:
             logging.warning(f"Không tìm thấy context nào cho query: '{query}'")
        else:
             logging.info(f"Đã tìm thấy {len(contexts)} context liên quan.")

        if not getattr(generator, 'client', None) and not getattr(generator, 'model', None):
             logging.error("Generator client/model không sẵn sàng (kiểm tra API key?). Không thể tạo câu trả lời.")
             return jsonify({"error": "Không thể kết nối đến dịch vụ sinh câu trả lời (vấn đề API key?)."}), 503

        logging.info("Bắt đầu quá trình Generate...")
        final_answer = generator.generate_response(query, contexts) 

        # Dịch câu trả lời sang tiếng Việt nếu ngôn ngữ đầu vào là tiếng Việt
        if detected_language == 'vi':
            logging.info("Dịch câu trả lời từ tiếng Anh sang tiếng Việt...")
            final_answer = GoogleTranslator(source='en', target='vi').translate(final_answer)

        logging.info(f"Câu trả lời được tạo: '{final_answer[:100]}...'") 
        return jsonify({"answer": final_answer})

    except Exception as e:
        logging.error(f"Đã xảy ra lỗi không mong muốn khi xử lý '/chat': {e}", exc_info=True)
        return jsonify({"error": "Đã xảy ra lỗi máy chủ nội bộ."}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PYTHON_API_PORT', 5001)) 
    logging.info(f"Flask server đang khởi động tại http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False) 