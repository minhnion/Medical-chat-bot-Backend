import express from "express";

import {
  getConversations,
  handleNewMessage
} from "../controllers/conversation.controller.js";

const router = express.Router();

router.post("/", getConversations);
router.post("/message", handleNewMessage)

export default router;
