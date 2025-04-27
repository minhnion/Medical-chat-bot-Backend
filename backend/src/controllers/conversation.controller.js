import Conversation from "../models/conversation.model.js";

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
