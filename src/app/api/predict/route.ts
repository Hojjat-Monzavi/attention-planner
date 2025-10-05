// app/api/predict/route.ts
import { NextResponse } from "next/server";
import * as ort from "onnxruntime-node";
import path from "path";

let session: ort.InferenceSession | null = null;
const modelPath = path.join(process.cwd(), "public", "models", "attention_planner.onnx");

async function getSession() {
  if (!session) {
    session = await ort.InferenceSession.create(modelPath, { executionProviders: ["cpu"] });
  }
  return session;
}

type Task = {
  priority: number;
  min_duration: number;
  max_duration: number;
  due_time: number;
};

function taskToVec(t: Task, currentTime = 0) {
  const avgDur = (t.min_duration + t.max_duration) / 2;
  const slack = Math.max(0, t.due_time - currentTime - avgDur);
  return [
    t.priority / 10,
    avgDur / 4,
    t.due_time / 30,
    slack / 20,
    currentTime / 20,
    (t.priority * avgDur) / 40
  ];
}

export async function POST(req: Request) {
  const body = await req.json();
  const tasks: Task[] = body.tasks ?? [];
  const rest_factor: number = Number(body.rest_factor ?? 1.0);
  const greedy: boolean = body.greedy ?? true;

  if (!tasks.length) return NextResponse.json({ error: "No tasks provided" }, { status: 400 });

  const sess = await getSession();

  const plan: number[] = [];
  const remaining = tasks.map((_, i) => i);
  let currentTime = 0;

  while (remaining.length > 0) {
    const feats = remaining.map(i => taskToVec(tasks[i], currentTime));
    const numTasks = feats.length;
    const flat = new Float32Array(feats.flat());
    const tensor = new ort.Tensor("float32", flat, [numTasks, 6]);

    // mask is float32 1.0 / 0.0
    const maskArr = new Float32Array(numTasks).fill(1.0);
    const maskTensor = new ort.Tensor("float32", maskArr, [numTasks]);

    // rest_factor as shape [1]
    const restTensor = new ort.Tensor("float32", Float32Array.from([rest_factor]), [1]);

    const feeds: Record<string, ort.Tensor> = {
      task_features: tensor,
      rest_factor: restTensor,
      mask: maskTensor
    };

    const results = await sess.run(feeds);
    const logits = (results.logits.data as Float32Array);
    // softmax numerically stable
    const maxLogit = Math.max(...Array.from(logits));
    const exps = logits.map(v => Math.exp(v - maxLogit));
    const sumexp = exps.reduce((a,b) => a+b, 0);
    const probs = exps.map(e => e / sumexp);

    let idxSelected = 0;
    if (greedy) {
      idxSelected = probs.indexOf(Math.max(...probs));
    } else {
      // sample
      const r = Math.random();
      let cum = 0;
      idxSelected = probs.findIndex(p => (cum += p) >= r);
      if (idxSelected < 0) idxSelected = probs.length - 1;
    }

    const selected = remaining[idxSelected];
    plan.push(selected);

    const t = tasks[selected];
    const avgDur = (t.min_duration + t.max_duration) / 2;
    currentTime += avgDur * rest_factor;

    remaining.splice(idxSelected, 1);
  }

  // return ordered indices and also human-readable plan (optional)
  const orderedTasks = plan.map(i => ({ index: i, task: tasks[i] }));
  return NextResponse.json({ plan, orderedTasks });
}
