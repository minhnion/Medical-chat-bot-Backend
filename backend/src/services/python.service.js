import { PythonShell } from "python-shell";
import path from "path";

//gọi embedder
export function embedAndIndex(id, text) {
  const options = {
    mode: "text",
    pythonPath: process.env.PYTHON_CMD || "python",
    scriptPath: path.resolve(process.cwd(), process.env.PYTHON_SERVICE_PATH),
    args: ["--id", id, "--text", text, "--index", process.env.FAISS_INDEX_PATH],
  };
  return new Promise((res, rej) => {
    PythonShell.run("embedder.py", options, (err, results) => {
      if (err) return rej(err);
      res(JSON.parse(results[0]));
    });
  });
}

//gọi retriever (chat)
export function retrieveAnswer(question) {
  const options = {
    mode: "text",
    pythonPath: process.env.PYTHON_CMD || "python",
    scriptPath: path.resolve(process.cwd(), process.env.PYTHON_SERVICE_PATH),
    args: ["--question", question, "--index", process.env.FAISS_INDEX_PATH, "--topk", "5"],
  };
  return new Promise((res, rej) => {
    PythonShell.run("retriever.py", options, (err, results) => {
      if (err) return rej(err);
      res(JSON.parse(results[0]));  
    });
  });
}