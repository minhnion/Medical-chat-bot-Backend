import mongoose from "mongoose";

const ConversationSchema = new mongoose.Schema(
  {
    description: {
      type: String,
      required: true,
    },
    patient: { type: String, required: true },
    doctor: { type: String, required: true },
  },
  {
    timestamps: true,
  }
);

const Conversation = mongoose.model("Conversation", ConversationSchema);

export default Conversation;
