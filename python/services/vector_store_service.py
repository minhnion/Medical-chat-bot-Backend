# Kết nối tới MongoDB.
# Đọc dữ liệu từ collection conversations.
# Chọn trường dữ liệu cần tạo embedding (ví dụ: description).
# Sử dụng một mô hình Sentence Transformer để chuyển đổi text thành vector.
# Tạo một index FAISS.
# Thêm các vector vào index FAISS.
# Lưu index FAISS và một bản đồ (mapping) từ vị trí trong index về lại _id của document gốc trong MongoDB để sau này có thể truy xuất.

import os
import faiss
import numpy as np
import pickle
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import logging
from urllib.parse import urlparse


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# .env file path
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'backend/.env')
load_dotenv(dotenv_path=dotenv_path)

# config
MONGO_URI = os.getenv("MONGODB_URI")
if not MONGO_URI:
    logging.error("MONGODB_URI không được đặt trong file .env!")
    raise ValueError("Thiếu MONGODB_URI")

parsed_uri = urlparse(MONGO_URI)
db_name_from_uri = parsed_uri.path.strip('/') if parsed_uri.path else None

db_name_from_env = os.getenv("DB_NAME")

FINAL_DB_NAME = None
if db_name_from_env:
    FINAL_DB_NAME = db_name_from_env
    logging.info(f"Sử dụng DB_NAME được chỉ định trong .env: {FINAL_DB_NAME}")
elif db_name_from_uri:
    FINAL_DB_NAME = db_name_from_uri
    logging.info(f"Sử dụng DB_NAME được lấy từ MONGODB_URI: {FINAL_DB_NAME}")
else:
    logging.error("Không thể xác định tên database. Vui lòng đặt DB_NAME trong .env hoặc đảm bảo MONGODB_URI chứa tên DB (ví dụ: ...mongodb.net/your_db_name).")
    raise ValueError("Không thể xác định tên database")

COLLECTION_NAME = "conversations"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
INDEX_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector_store', 'faiss_index.bin')
MAPPING_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector_store', 'id_mapping.pkl')

os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)


class VectorStoreService:
    def __init__(self, mongo_uri=MONGO_URI, db_name=FINAL_DB_NAME, collection_name=COLLECTION_NAME, model_name=EMBEDDING_MODEL):
        logging.info("Khởi tạo VectorStoreService...")
        
        self.mongo_uri = mongo_uri
        self.db_name = db_name 
        self.collection_name = collection_name
        try:
            self.client = MongoClient(self.mongo_uri)
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            logging.info(f"Kết nối thành công tới MongoDB: DB='{self.db_name}', Collection='{self.collection_name}'")
        except Exception as e:
            logging.error(f"Lỗi kết nối MongoDB: {e}")
            if "query timed out" in str(e).lower():
                logging.error("Lỗi timeout - kiểm tra cấu hình mạng/firewall tới MongoDB Atlas.")
            elif "authentication failed" in str(e).lower():
                logging.error("Lỗi xác thực - kiểm tra username/password trong MONGODB_URI.")
            raise ConnectionError(f"Không thể kết nối tới MongoDB (DB: {self.db_name}): {e}")

        logging.info(f"Đang tải model embedding: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logging.info(f"Model {model_name} đã tải xong. Kích thước vector: {self.embedding_dim}")
        except Exception as e:
            logging.error(f"Lỗi khi tải model embedding: {e}")
            raise ValueError(f"Không thể tải model embedding '{model_name}': {e}")

    def _fetch_data(self):
        logging.info(f"Đang lấy dữ liệu từ collection '{self.collection_name}'...")
        try:
            documents = list(self.collection.find({}, {"_id": 1, "Description": 1, "Doctor": 1}))
            if not documents:
                logging.warning("Không tìm thấy tài liệu nào.")
                return [], []

            ids = [str(doc["_id"]) for doc in documents]
            texts_to_embed = []
            for doc in documents:
                question = doc.get("Description", "")
                answer = doc.get("Doctor", "")
                combined_text = f"Câu hỏi: {question}\nTrả lời: {answer}"
                texts_to_embed.append(combined_text)

            logging.info(f"Đã chuẩn bị {len(texts_to_embed)} đoạn text để embed.")
            return ids, texts_to_embed 
        except Exception as e:
            logging.error(f"Lỗi khi truy vấn hoặc chuẩn bị dữ liệu MongoDB: {e}")
            return [], []

    def build_and_save_index(self, index_path=INDEX_PATH, mapping_path=MAPPING_PATH):
        mongo_ids, texts_to_embed  = self._fetch_data()

        if not texts_to_embed:
            logging.warning("Không có dữ liệu để tạo index. Bỏ qua.")
            return

        logging.info(f"Đang tạo embeddings cho {len(texts_to_embed)} mô tả...")
        try:
            embeddings = self.model.encode(texts_to_embed, show_progress_bar=True, convert_to_numpy=True)
            embeddings = embeddings.astype('float32')
            logging.info(f"Đã tạo xong {embeddings.shape[0]} embeddings với kích thước {embeddings.shape[1]}.")
        except Exception as e:
            logging.error(f"Lỗi trong quá trình tạo embedding: {e}")
            return 

        logging.info(f"Đang xây dựng FAISS index (IndexFlatL2)...")
        try:
            index = faiss.IndexFlatL2(self.embedding_dim)
            if embeddings.shape[0] > 0:
                 index.add(embeddings)
                 logging.info(f"Đã thêm {index.ntotal} vector vào index FAISS.")
            else:
                 logging.warning("Không có embeddings nào được tạo để thêm vào index.")

            logging.info(f"Đang lưu index FAISS vào: {index_path}")
            faiss.write_index(index, index_path)
            logging.info("Lưu index FAISS thành công.")

            id_mapping = {i: mongo_id for i, mongo_id in enumerate(mongo_ids)}
            logging.info(f"Đang lưu id mapping vào: {mapping_path}")
            with open(mapping_path, 'wb') as f:
                pickle.dump(id_mapping, f)
            logging.info("Lưu id mapping thành công.")

        except faiss.FaissException as e:
             logging.error(f"Lỗi FAISS: {e}")
        except IOError as e:
             logging.error(f"Lỗi I/O khi lưu file index/mapping: {e}")
        except Exception as e:
            logging.error(f"Lỗi không mong muốn khi xây dựng/lưu index: {e}")
            return

        logging.info("Hoàn tất quá trình xây dựng và lưu trữ vector store.")


    def close_connection(self):
        """Đóng kết nối MongoDB."""
        if hasattr(self, 'client') and self.client:
            self.client.close()
            logging.info("Đã đóng kết nối MongoDB.")

if __name__ == "__main__":
    logging.info("Bắt đầu quá trình tạo Vector Store...")
    service = None
    try:
        service = VectorStoreService() 
        service.build_and_save_index()
        logging.info("Vector Store đã được tạo/cập nhật thành công.")
    except (ConnectionError, ValueError, TypeError) as e: 
        logging.error(f"Lỗi cấu hình, kết nối hoặc dữ liệu: {e}")
    except Exception as e:
        logging.error(f"Đã xảy ra lỗi không mong muốn trong quá trình chính: {e}", exc_info=True)
    finally:
        if service:
            service.close_connection()