# Targer: 
# Nhận một câu hỏi (query) từ người dùng.
# Sử dụng cùng một model embedding (all-MiniLM-L6-v2 hoặc model bạn đã chọn) để tạo vector cho câu hỏi đó.
# Tải index FAISS (faiss_index.bin) và bản đồ ID (id_mapping.pkl) mà bạn vừa tạo.
# Tìm kiếm trong index FAISS để tìm ra các đoạn mô tả (description trong collection Conversation) có nội dung gần giống/liên quan nhất đến câu hỏi.
# Trả về các _id hoặc nội dung của các đoạn mô tả liên quan đó. Đây chính là "ngữ cảnh" (context) mà chúng ta sẽ cung cấp cho LLM ở bước sau.

import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import logging
from pymongo import MongoClient 
from bson import ObjectId 

# logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# .env file path
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'backend/.env')
load_dotenv(dotenv_path=dotenv_path)

# config 
MONGO_URI = os.getenv("MONGODB_URI")
try:
    from urllib.parse import urlparse
    parsed_uri = urlparse(MONGO_URI)
    db_name_from_uri = parsed_uri.path.strip('/') if parsed_uri.path else None
except (ImportError, TypeError): 
     logging.warning("Không thể parse MONGODB_URI để lấy tên DB.")
     db_name_from_uri = None
db_name_from_env = os.getenv("DB_NAME")
FINAL_DB_NAME = db_name_from_env or db_name_from_uri
if not FINAL_DB_NAME:
    logging.warning("Không xác định được tên DB, sẽ không thể fetch context.")

COLLECTION_NAME = "conversations"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2' 

INDEX_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector_store', 'faiss_index.bin')
MAPPING_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector_store', 'id_mapping.pkl')


class RetrieverService:
    def __init__(self, index_path=INDEX_PATH, mapping_path=MAPPING_PATH, model_name=EMBEDDING_MODEL, mongo_uri=MONGO_URI, db_name=FINAL_DB_NAME, collection_name=COLLECTION_NAME):
        logging.info("Khởi tạo RetrieverService...")
        self.index_path = index_path
        self.mapping_path = mapping_path
        self.model_name = model_name
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None

        try:
            logging.info(f"Đang tải model embedding: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logging.info(f"Model {self.model_name} đã tải xong.")
        except Exception as e:
            logging.error(f"Lỗi khi tải model embedding '{self.model_name}': {e}")
            raise ValueError(f"Không thể tải model embedding: {e}")

        try:
            logging.info(f"Đang tải FAISS index từ: {self.index_path}")
            if not os.path.exists(self.index_path):
                 raise FileNotFoundError(f"Không tìm thấy file index FAISS tại: {self.index_path}")
            self.index = faiss.read_index(self.index_path)
            logging.info(f"Tải FAISS index thành công. Tổng số vector: {self.index.ntotal}")
        except (faiss.FaissException, FileNotFoundError, Exception) as e:
            logging.error(f"Lỗi khi tải FAISS index: {e}")
            raise RuntimeError(f"Không thể tải index FAISS: {e}")

        try:
            logging.info(f"Đang tải ID mapping từ: {self.mapping_path}")
            if not os.path.exists(self.mapping_path):
                raise FileNotFoundError(f"Không tìm thấy file mapping tại: {self.mapping_path}")
            with open(self.mapping_path, 'rb') as f:
                self.id_mapping = pickle.load(f)
            logging.info(f"Tải ID mapping thành công. Số lượng mapping: {len(self.id_mapping)}")
            # Kiểm tra sơ bộ xem số mapping có khớp với số vector không
            if self.index.ntotal != len(self.id_mapping):
                logging.warning(f"Số lượng vector trong index ({self.index.ntotal}) không khớp với số lượng ID trong mapping ({len(self.id_mapping)}). Có thể có vấn đề.")
        except (FileNotFoundError, pickle.PickleError, Exception) as e:
            logging.error(f"Lỗi khi tải ID mapping: {e}")
            raise RuntimeError(f"Không thể tải ID mapping: {e}")

        if self.mongo_uri and self.db_name:
            try:
                logging.info(f"Đang kết nối tới MongoDB để fetch context: DB='{self.db_name}'")
                self.client = MongoClient(self.mongo_uri)
                self.db = self.client[self.db_name]
                self.collection = self.db[self.collection_name]
                # Kiểm tra kết nối nhanh
                self.client.admin.command('ping')
                logging.info("Kết nối MongoDB thành công.")
            except Exception as e:
                logging.error(f"Lỗi kết nối MongoDB (để fetch context): {e}. Chức năng fetch context sẽ không hoạt động.")
                self.client = None # Đặt lại để không cố sử dụng kết nối lỗi
                self.db = None
                self.collection = None
        else:
             logging.warning("Thiếu MONGO_URI hoặc DB_NAME, sẽ không fetch context từ MongoDB.")


    def retrieve(self, query: str, top_k: int = 5, fetch_context: bool = True) -> list:
        if not query:
            logging.warning("Query rỗng, không thực hiện tìm kiếm.")
            return []
        if self.index.ntotal == 0:
            logging.warning("Index FAISS rỗng, không có gì để tìm kiếm.")
            return []

        logging.info(f"Đang tạo embedding cho query: '{query[:50]}...'") # Log 50 ký tự đầu
        try:
            query_embedding = self.model.encode([query], convert_to_numpy=True).astype('float32')
        except Exception as e:
             logging.error(f"Lỗi khi tạo embedding cho query: {e}")
             return [] 

        logging.info(f"Đang tìm kiếm {top_k} kết quả gần nhất trong FAISS index...")
        try:
            distances, indices = self.index.search(query_embedding, top_k)

            results = []
            logging.info(f"FAISS indices tìm thấy: {indices[0]}")
            
            for i, idx in enumerate(indices[0]):
                mongo_id = None 
                if idx != -1:
                    mongo_id = self.id_mapping.get(idx) 
                    score = 1.0 - distances[0][i]

                    logging.info(f"Đang xử lý kết quả {i+1}: FAISS Index={idx}, Score={score:.4f}")
                    if mongo_id:
                        logging.info(f"  -> MongoDB ID tương ứng (từ mapping): {mongo_id}")
                        result_item = {'id': mongo_id, 'score': float(score)}

                        if fetch_context and self.collection is not None:
                            logging.info(f"  -> Đang tìm kiếm document _id='{mongo_id}' trong MongoDB...")
                            doc = None 
                            try:
                                doc = self.collection.find_one({"_id": ObjectId(mongo_id)}, {"Description": 1, "Doctor": 1})
                                
                                if doc:
                                    logging.info(f"  -> Tìm thấy document!")
                                    doctor_answer = doc.get('Doctor')
                                    question = doc.get('Description') 

                                    if doctor_answer:
                                        logging.info(f"    -> Lấy 'Doctor' làm context: {doctor_answer[:100]}...")
                                        result_item['context'] = doctor_answer 
                                    elif question: 
                                        logging.warning(f"    -> Không có 'Doctor', dùng 'Description' làm context: {question[:100]}...")
                                        result_item['context'] = question
                                    else:
                                        logging.warning(f"    -> KHÔNG tìm thấy cả 'Doctor' và 'Description'!")
                                        result_item['context'] = None
                                
                            except Exception as e:
                                logging.error(f"  -> Lỗi khi truy vấn MongoDB cho _id={mongo_id}: {e}", exc_info=True) # Thêm exc_info
                                result_item['context'] = None

                        results.append(result_item)
                    else:
                         logging.warning(f"  -> Không tìm thấy MongoDB ID cho FAISS index {idx} trong mapping.")
                else:
                    logging.info(f"Đang xử lý kết quả {i+1}: FAISS Index={idx} (Không hợp lệ, bỏ qua).")

            logging.info(f"Tổng số kết quả hợp lệ được xử lý (trước khi lọc context): {len(results)}")
            return results 

        except Exception as e: 
            logging.error(f"Lỗi khi tìm kiếm trong FAISS index hoặc xử lý kết quả: {e}", exc_info=True)
            return []

    def close_connection(self):
        """Đóng kết nối MongoDB nếu có."""
        if self.client:
            self.client.close()
            logging.info("Đã đóng kết nối MongoDB (Retriever).")


if __name__ == "__main__":
    logging.info("Bắt đầu kiểm tra RetrieverService...")
    retriever = None
    try:
        retriever = RetrieverService()

        test_query = "Triệu chứng đau đầu và sốt nhẹ là bệnh gì?"
        print(f"\n--- Thử tìm kiếm cho query: '{test_query}' ---")

        results_with_context = retriever.retrieve(test_query, top_k=3, fetch_context=True)
        print("\nKết quả (với context):")
        if results_with_context:
            for res in results_with_context:
                print(f"  ID: {res['id']}, Score: {res['score']:.4f}, Context: {res.get('context', 'N/A')[:100]}...") 
        else:
            print("  Không tìm thấy kết quả.")

        results_ids_only = retriever.retrieve(test_query, top_k=2, fetch_context=False)
        print("\nKết quả (chỉ ID):")
        if results_ids_only:
            for res in results_ids_only:
                 print(f"  ID: {res['id']}, Score: {res['score']:.4f}")
        else:
             print("  Không tìm thấy kết quả.")

    except (RuntimeError, ValueError, FileNotFoundError) as e:
        logging.error(f"Lỗi khi khởi tạo hoặc chạy RetrieverService: {e}")
    except Exception as e:
        logging.error(f"Lỗi không mong muốn: {e}", exc_info=True)
    finally:
        if retriever:
            retriever.close_connection()
        logging.info("Kết thúc kiểm tra RetrieverService.")