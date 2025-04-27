import "dotenv/config";
import express from "express";
import cors from "cors";
import morgan from "morgan";
import mongoose from "mongoose";
import conversationRoute from "./routes/conversation.route.js";

const app = express();
app.use(express.json());

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on ${PORT}`));

//middleware
app.use(cors());
app.use(morgan("dev"));

//Routes
app.use("/api/conversations", conversationRoute);

mongoose
  .connect(process.env.MONGODB_URI, {
    maxPoolSize: 50,
  })
  .then(() => console.log("MongoDB connected"))
  .catch((err) => console.error("MongoDB connection error:", err));
