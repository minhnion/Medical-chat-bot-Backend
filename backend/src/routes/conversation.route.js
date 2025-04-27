import express from "express";

import {
  getConversations
} from "../controllers/conversation.controller.js";

const router = express.Router();

router.post("/", getConversations);


export default router;
