import { existsSync } from "node:fs";
import path from "node:path";

export const repo = path.resolve(process.cwd(), "..");
export const features = path.join(repo, "processed", "features", "year=2026");
export const python = process.env.PYTHON_PATH || [
  path.join(repo, "env", "python.exe"),
  path.join(repo, "env", "Scripts", "python.exe"),
  path.join(repo, "env", "bin", "python"),
].find(existsSync) || "python";
export const pythonOptions = {
  cwd: repo,
  windowsHide: true,
  env: { ...process.env, PYTHONIOENCODING: "utf-8" },
};
