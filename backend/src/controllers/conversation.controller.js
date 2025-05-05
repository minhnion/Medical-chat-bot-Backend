import Conversation from "../models/conversation.model.js";
import { pythonService } from "../services/python.service.js";

export async function getConversations(req, res) {
  try {
    const { page = 1, limit = 10 } = req.body;
    const pageNum = Math.max(1, parseInt(page, 10));
    const limitNum = Math.max(1, parseInt(limit, 10));

    const skip = (pageNum - 1) * limitNum;

    const [ total, conversations ] = await Promise.all([
      Conversation.countDocuments({}),                             
      Conversation
        .find({})
        .sort({ createdAt: -1 })                                   
        .skip(skip)
        .limit(limitNum)
        .lean()
    ]);

    const totalPages = Math.ceil(total / limitNum);

    return res.json({
      total,
      totalPages,
      page: pageNum,
      limit: limitNum,
      data: conversations
    });
  } catch (err) {
    console.error("getConversations error:", err);
    return res.status(500).json({ error: "Internal server error" });
  }
}


export const handleNewMessage = async (req, res) => {
  const { query } = req.body;

  if (!query) {
    return res.status(400).json({ error: "Thiếu 'query' trong yêu cầu." });
  }

  try {
    console.log(`[Controller] Nhận query: "${query}"`);
    const ragAnswer = await pythonService.getRagResponse(query);

    console.log(`[Controller] Nhận được câu trả lời RAG: "${ragAnswer.substring(0,100)}..."`);

    res.status(200).json({
       answer: ragAnswer,
    });

  } catch (error) {
    console.error("[Controller] Lỗi khi xử lý tin nhắn:", error.message);

    if (error.message.startsWith("Lỗi từ Python Service:") || error.message.startsWith("Không thể kết nối")) {
       res.status(503).json({ error: `Không thể nhận câu trả lời từ dịch vụ AI: ${error.message}` }); // 503 Service Unavailable
    } else if (error.message.startsWith("Phản hồi không hợp lệ")) {
        res.status(502).json({ error: `Dịch vụ AI trả về phản hồi không mong muốn.` }); // 502 Bad Gateway
    }
    else {
       res.status(500).json({ error: "Lỗi máy chủ nội bộ khi xử lý yêu cầu." });
    }
  }
};