"use client";

import { useEffect, useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Trash2, Download, PlusCircle, Loader2, Calendar, Clock, Settings } from "lucide-react";
import * as ort from "onnxruntime-web";

type Task = {
  name: string;
  priority: number;
  min_duration: number;
  max_duration: number;
  due_date: string;
  available_hours: number;
};

const LS_TASKS_KEY = "planner_tasks_v2";
const LS_SCHEDULE_KEY = "planner_schedule_v1";
const MODEL_CACHE_KEY = "onnx_model_cache";
const MODEL_URL = "/models/attention_planner.onnx";

export default function Page() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [name, setName] = useState("");
  const [priority, setPriority] = useState(5);
  const [minD, setMinD] = useState(1.0);
  const [maxD, setMaxD] = useState(1.5);
  const [dueDate, setDueDate] = useState("");
  const [dailyHours, setDailyHours] = useState(8);
  const [weekendHours, setWeekendHours] = useState(4);
  const [rest, setRest] = useState(1.2);
  const [plan, setPlan] = useState<number[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [modelError, setModelError] = useState<string | null>(null);
  
  const sessionRef = useRef<ort.InferenceSession | null>(null);

  useEffect(() => {
    async function loadModel() {
      try {
        // Try to load from cache first
        const cache = await caches.open(MODEL_CACHE_KEY);
        let modelData: ArrayBuffer;

        const cachedResponse = await cache.match(MODEL_URL);
        
        if (cachedResponse) {
          // Load from cache
          modelData = await cachedResponse.arrayBuffer();
        } else {
          // Download and cache
          const response = await fetch(MODEL_URL);
          if (!response.ok) throw new Error("Failed to fetch model");
          
          // Clone response for caching
          await cache.put(MODEL_URL, response.clone());
          modelData = await response.arrayBuffer();
        }

        // Create session from buffer
        const session = await ort.InferenceSession.create(modelData, {
          executionProviders: ["wasm"]
        });
        
        sessionRef.current = session;
        setModelReady(true);
      } catch (err) {
        console.error("Failed to load ONNX model:", err);
        setModelError("Failed to load AI model. Please check your connection and refresh.");
      }
    }
    loadModel();
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    
    const rawTasks = localStorage.getItem(LS_TASKS_KEY);
    if (rawTasks) {
      try {
        const parsed = JSON.parse(rawTasks);
        if (Array.isArray(parsed)) {
          setTasks(parsed.map((t: any, i: number) => ({
            name: t.name || `Task ${i + 1}`,
            priority: t.priority ?? 5,
            min_duration: t.min_duration ?? 1,
            max_duration: t.max_duration ?? 1.5,
            due_date: t.due_date || new Date().toISOString().split('T')[0],
            available_hours: t.available_hours ?? 0,
          })));
        }
      } catch (err) {
        console.error("Failed to parse tasks", err);
      }
    }

    const rawSchedule = localStorage.getItem(LS_SCHEDULE_KEY);
    if (rawSchedule) {
      try {
        const parsed = JSON.parse(rawSchedule);
        setDailyHours(parsed.dailyHours ?? 8);
        setWeekendHours(parsed.weekendHours ?? 4);
      } catch (err) {
        console.error("Failed to parse schedule", err);
      }
    }

    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1);
    setDueDate(tomorrow.toISOString().split('T')[0]);
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      localStorage.setItem(LS_TASKS_KEY, JSON.stringify(tasks));
    } catch (e) {
      console.error("Failed saving tasks", e);
    }
  }, [tasks]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      localStorage.setItem(LS_SCHEDULE_KEY, JSON.stringify({ dailyHours, weekendHours }));
    } catch (e) {
      console.error("Failed saving schedule", e);
    }
  }, [dailyHours, weekendHours]);

  function calculateAvailableHours(dueDate: string): number {
    const now = new Date();
    const deadline = new Date(`${dueDate}T17:00`); // Default to 5 PM
    
    if (deadline <= now) return 0;

    let totalHours = 0;
    let current = new Date(now);
    current.setHours(0, 0, 0, 0);

    while (current < deadline) {
      const isWeekend = current.getDay() === 0 || current.getDay() === 6;
      const dayHours = isWeekend ? weekendHours : dailyHours;
      
      if (current.toDateString() === deadline.toDateString()) {
        const deadlineHour = deadline.getHours() + deadline.getMinutes() / 60;
        const workdayStart = 9;
        const availableToday = Math.max(0, Math.min(dayHours, deadlineHour - workdayStart));
        totalHours += availableToday;
      } else if (current > now) {
        totalHours += dayHours;
      } else {
        const nowHour = now.getHours() + now.getMinutes() / 60;
        const workdayStart = 9;
        const workdayEnd = workdayStart + dayHours;
        const remainingToday = Math.max(0, workdayEnd - nowHour);
        totalHours += remainingToday;
      }
      
      current.setDate(current.getDate() + 1);
    }

    return Math.max(0, Math.round(totalHours * 10) / 10);
  }

  function taskToVec(t: { priority: number; min_duration: number; max_duration: number; available_hours: number }, currentTime = 0): number[] {
    const avgDur = (t.min_duration + t.max_duration) / 2;
    const slack = Math.max(0, t.available_hours - currentTime - avgDur);
    return [
      t.priority / 10,
      avgDur / 4,
      t.available_hours / 30,
      slack / 20,
      currentTime / 20,
      (t.priority * avgDur) / 40
    ];
  }

  async function getPlan(greedy = true) {
    if (!sessionRef.current) {
      alert("Model is still loading. Please wait a moment.");
      return;
    }

    setLoading(true);
    setPlan(null);

    try {
      const apiTasks = tasks.map(({ available_hours, ...rest }) => ({
        ...rest,
        due_time: available_hours
      }));

      const planIndices: number[] = [];
      const remaining = apiTasks.map((_, i) => i);
      let currentTime = 0;

      while (remaining.length > 0) {
        const feats = remaining.map(i => taskToVec({
          priority: apiTasks[i].priority,
          min_duration: apiTasks[i].min_duration,
          max_duration: apiTasks[i].max_duration,
          available_hours: apiTasks[i].due_time
        }, currentTime));
        
        const numTasks = feats.length;
        const flat = new Float32Array(feats.flat());
        const tensor = new ort.Tensor("float32", flat, [numTasks, 6]);

        const maskArr = new Float32Array(numTasks).fill(1.0);
        const maskTensor = new ort.Tensor("float32", maskArr, [numTasks]);

        const restTensor = new ort.Tensor("float32", Float32Array.from([rest]), [1]);

        const feeds: Record<string, ort.Tensor> = {
          task_features: tensor,
          rest_factor: restTensor,
          mask: maskTensor
        };

        const results = await sessionRef.current.run(feeds);
        const logits = results.logits.data as Float32Array;
        
        const maxLogit = Math.max(...Array.from(logits));
        const exps = Array.from(logits).map(v => Math.exp(v - maxLogit));
        const sumexp = exps.reduce((a, b) => a + b, 0);
        const probs = exps.map(e => e / sumexp);

        let idxSelected = 0;
        if (greedy) {
          idxSelected = probs.indexOf(Math.max(...probs));
        } else {
          const r = Math.random();
          let cum = 0;
          idxSelected = probs.findIndex(p => (cum += p) >= r);
          if (idxSelected < 0) idxSelected = probs.length - 1;
        }

        const selected = remaining[idxSelected];
        planIndices.push(selected);

        const t = apiTasks[selected];
        const avgDur = (t.min_duration + t.max_duration) / 2;
        currentTime += avgDur * rest;

        remaining.splice(idxSelected, 1);
      }

      setPlan(planIndices);
    } catch (err) {
      console.error("Planning error:", err);
      alert("Failed to generate plan. Please try again.");
    }

    setLoading(false);
  }

  function addTask() {
    const newName = name?.trim() || `Task ${tasks.length + 1}`;
    const availableHours = calculateAvailableHours(dueDate);
    
    setTasks((prev) => [
      ...prev,
      { 
        name: newName, 
        priority: priority || 5, 
        min_duration: minD || 1, 
        max_duration: maxD || 1.5, 
        due_date: dueDate,
        available_hours: availableHours
      },
    ]);
    setName("");
    setPriority(5);
    setMinD(1.0);
    setMaxD(1.5);
    
    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1);
    setDueDate(tomorrow.toISOString().split('T')[0]);
  }

  function clearTasks() {
    setTasks([]);
    setPlan(null);
  }

  async function clearCache() {
    try {
      const cache = await caches.open(MODEL_CACHE_KEY);
      await cache.delete(MODEL_URL);
      alert("Cache cleared! Please refresh the page to reload the model.");
    } catch (err) {
      console.error("Failed to clear cache:", err);
    }
  }

  function exportTasks() {
    const blob = new Blob([JSON.stringify(tasks, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "tasks.json";
    a.click();
    URL.revokeObjectURL(url);
  }

  function formatDueDate(dateStr: string): string {
    const date = new Date(dateStr);
    const today = new Date();
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);
    
    if (date.toDateString() === today.toDateString()) return "Today";
    if (date.toDateString() === tomorrow.toDateString()) return "Tomorrow";
    
    const diff = Math.ceil((date.getTime() - today.getTime()) / (1000 * 60 * 60 * 24));
    if (diff > 0 && diff < 7) return `In ${diff} days`;
    
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  }

  if (modelError) {
    return (
      <main className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50 flex items-center justify-center p-4">
        <Card className="max-w-md">
          <CardHeader>
            <CardTitle className="text-red-600">Error Loading Model</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-slate-700">{modelError}</p>
            <div className="flex gap-2">
              <Button onClick={() => window.location.reload()} className="flex-1">
                Retry
              </Button>
              <Button variant="outline" onClick={clearCache} className="flex-1">
                Clear Cache
              </Button>
            </div>
          </CardContent>
        </Card>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50 p-4 md:p-8 font-sans text-slate-800">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="max-w-4xl mx-auto space-y-6"
      >
        <div className="text-center space-y-2 mb-8">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
            Intelligent Task Planner
          </h1>
          <p className="text-slate-600">Optimize your schedule with AI-powered planning</p>
          {modelReady ? (
            <div className="inline-flex items-center gap-2 px-3 py-1 bg-green-100 text-green-700 rounded-full text-xs font-medium">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              AI Model Ready
            </div>
          ) : (
            <div className="inline-flex items-center gap-2 px-3 py-1 bg-amber-100 text-amber-700 rounded-full text-xs font-medium">
              <Loader2 className="w-3 h-3 animate-spin" />
              Loading AI Model...
            </div>
          )}
        </div>

        <Card className="shadow-lg border-indigo-100">
          <CardHeader className="bg-gradient-to-r from-indigo-50 to-purple-50">
            <CardTitle className="text-xl font-semibold flex items-center gap-2">
              <PlusCircle className="text-indigo-600" size={20} />
              Add New Task
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-6">
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="md:col-span-2">
                  <Label className="text-sm font-medium">Task Name</Label>
                  <Input 
                    value={name} 
                    onChange={(e) => setName(e.target.value)} 
                    placeholder={`e.g., Finish project proposal`}
                    className="mt-1"
                  />
                </div>

                <div className="md:col-span-2">
                  <Label className="text-sm font-medium flex items-center gap-2">
                    <Calendar size={14} /> Due Date
                  </Label>
                  <Input 
                    type="date" 
                    value={dueDate} 
                    onChange={(e) => setDueDate(e.target.value)}
                    className="mt-1"
                  />
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4">
                <div>
                  <Label className="text-sm font-medium">Priority</Label>
                  <Input 
                    type="number" 
                    min={1} 
                    max={10} 
                    value={priority || ""}
                    onChange={(e) => setPriority(e.target.value === "" ? 0 : Number(e.target.value))}
                    placeholder="5"
                    className="mt-1"
                  />
                  <p className="text-xs text-slate-500 mt-1">1-10 scale</p>
                </div>

                <div>
                  <Label className="text-sm font-medium">Min Duration</Label>
                  <Input 
                    type="number" 
                    step="0.5"
                    min={0}
                    value={minD || ""}
                    onChange={(e) => setMinD(e.target.value === "" ? 0 : Number(e.target.value))}
                    placeholder="1.0"
                    className="mt-1"
                  />
                  <p className="text-xs text-slate-500 mt-1">hours</p>
                </div>

                <div>
                  <Label className="text-sm font-medium">Max Duration</Label>
                  <Input 
                    type="number" 
                    step="0.5"
                    min={0}
                    value={maxD || ""}
                    onChange={(e) => setMaxD(e.target.value === "" ? 0 : Number(e.target.value))}
                    placeholder="1.5"
                    className="mt-1"
                  />
                  <p className="text-xs text-slate-500 mt-1">hours</p>
                </div>
              </div>

              {dueDate && (
                <div className="p-3 bg-indigo-50 rounded-lg border border-indigo-200">
                  <p className="text-sm text-indigo-900">
                    <strong>Available time:</strong> {calculateAvailableHours(dueDate).toFixed(1)} hours 
                    <span className="text-indigo-600"> · {formatDueDate(dueDate)}</span>
                  </p>
                </div>
              )}

              <div className="flex flex-wrap gap-3 pt-2">
                <Button 
                  onClick={addTask} 
                  className="bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700"
                  disabled={!dueDate}
                >
                  <PlusCircle size={16} className="mr-2" /> Add Task
                </Button>
                <Button variant="outline" onClick={() => setShowSettings(!showSettings)}>
                  <Settings size={16} className="mr-2" /> Settings
                </Button>
                <Button variant="outline" onClick={exportTasks}>
                  <Download size={16} className="mr-2" /> Export
                </Button>
                {tasks.length > 0 && (
                  <Button variant="destructive" onClick={clearTasks}>Clear All</Button>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        <AnimatePresence>
          {showSettings && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
            >
              <Card className="shadow-md border-slate-200">
                <CardHeader>
                  <CardTitle className="text-lg font-semibold">Schedule Settings</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <Label className="text-sm font-medium">Weekday Hours</Label>
                      <Input 
                        type="number" 
                        step="0.5"
                        min={0}
                        value={dailyHours || ""}
                        onChange={(e) => setDailyHours(e.target.value === "" ? 0 : Number(e.target.value))}
                        placeholder="8"
                        className="mt-1"
                      />
                      <p className="text-xs text-slate-500 mt-1">Mon-Fri available hours</p>
                    </div>

                    <div>
                      <Label className="text-sm font-medium">Weekend Hours</Label>
                      <Input 
                        type="number" 
                        step="0.5"
                        min={0}
                        value={weekendHours || ""}
                        onChange={(e) => setWeekendHours(e.target.value === "" ? 0 : Number(e.target.value))}
                        placeholder="4"
                        className="mt-1"
                      />
                      <p className="text-xs text-slate-500 mt-1">Sat-Sun available hours</p>
                    </div>

                    <div>
                      <Label className="text-sm font-medium">Rest Factor</Label>
                      <Input 
                        type="number" 
                        step="0.1"
                        min={0}
                        value={rest || ""}
                        onChange={(e) => setRest(e.target.value === "" ? 0 : Number(e.target.value))}
                        placeholder="1.2"
                        className="mt-1"
                      />
                      <p className="text-xs text-slate-500 mt-1">Planning buffer multiplier</p>
                    </div>
                  </div>
                  
                  <div className="mt-4 pt-4 border-t">
                    <Button variant="outline" size="sm" onClick={clearCache}>
                      Clear Model Cache
                    </Button>
                    <p className="text-xs text-slate-500 mt-2">
                      Clear cached model data to force re-download
                    </p>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )}
        </AnimatePresence>

        <Card className="shadow-md">
          <CardHeader>
            <CardTitle className="text-lg font-semibold">Your Tasks ({tasks.length})</CardTitle>
          </CardHeader>
          <CardContent>
            {tasks.length === 0 ? (
              <div className="text-center py-12">
                <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Calendar className="text-slate-400" size={32} />
                </div>
                <p className="text-slate-500">No tasks yet. Add your first task to get started!</p>
              </div>
            ) : (
              <div className="space-y-3">
                {tasks.map((t, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.2, delay: i * 0.05 }}
                    className="group p-4 rounded-lg border border-slate-200 bg-white hover:border-indigo-300 hover:shadow-md transition-all"
                  >
                    <div className="flex items-start gap-4">
                      <div className="w-12 h-12 flex-shrink-0 flex items-center justify-center rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 text-white font-bold text-lg">
                        {i + 1}
                      </div>
                      <div className="flex-1 min-w-0">
                        <h3 className="font-semibold text-slate-900 mb-2">{t.name}</h3>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs text-slate-600">
                          <div>
                            <span className="font-medium">Priority:</span> {t.priority}/10
                          </div>
                          <div>
                            <span className="font-medium">Duration:</span> {t.min_duration}-{t.max_duration}h
                          </div>
                          <div>
                            <span className="font-medium">Due:</span> {formatDueDate(t.due_date)}
                          </div>
                          <div>
                            <span className="font-medium">Available:</span> {t.available_hours.toFixed(1)}h
                          </div>
                        </div>
                      </div>
                      <Button 
                        size="icon" 
                        variant="ghost" 
                        className="opacity-0 group-hover:opacity-100 transition-opacity"
                        onClick={() => setTasks(prev => prev.filter((_, idx) => idx !== i))}
                      >
                        <Trash2 size={16} className="text-red-500" />
                      </Button>
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {tasks.length > 0 && (
          <div className="flex flex-col items-center gap-3">
            {!modelReady && (
              <div className="text-sm text-amber-600 flex items-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin" />
                Waiting for AI model to load...
              </div>
            )}
            <div className="flex gap-4">
              <Button 
                onClick={() => getPlan(true)} 
                disabled={loading || !modelReady}
                size="lg"
                className="bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 px-8"
              >
                {loading ? (
                  <>
                    <Loader2 className="animate-spin mr-2" size={18} />
                    Generating...
                  </>
                ) : (
                  "Generate Optimal Plan"
                )}
              </Button>
              <Button 
                variant="outline" 
                onClick={() => getPlan(false)} 
                disabled={loading || !modelReady}
                size="lg"
              >
                Sample Plan
              </Button>
            </div>
          </div>
        )}

        <AnimatePresence>
          {plan && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
            >
              <Card className="shadow-lg border-green-200 bg-gradient-to-br from-green-50 to-emerald-50">
                <CardHeader>
                  <CardTitle className="text-xl font-semibold text-green-900 flex items-center gap-2">
                    <div className="w-8 h-8 bg-green-600 rounded-full flex items-center justify-center text-white text-sm">✓</div>
                    Your Optimized Plan
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {plan.map((idx, pos) => {
                      const t = tasks[idx];
                      if (!t) return null;
                      
                      const avg = ((t.min_duration + t.max_duration) / 2).toFixed(1);

                      return (
                        <motion.div
                          key={pos}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ duration: 0.3, delay: pos * 0.08 }}
                          className="flex items-center gap-4 p-4 rounded-lg bg-white border border-green-200 shadow-sm"
                        >
                          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-green-500 to-emerald-600 text-white flex items-center justify-center font-bold">
                            {pos + 1}
                          </div>
                          <div className="flex-1 min-w-0">
                            <h4 className="font-semibold text-slate-900">{t.name}</h4>
                            <div className="flex flex-wrap gap-3 mt-1 text-xs text-slate-600">
                              <span>Est. {avg}h</span>
                              <span>·</span>
                              <span>Priority {t.priority}</span>
                              <span>·</span>
                              <span>Due {formatDueDate(t.due_date)}</span>
                            </div>
                          </div>
                        </motion.div>
                      );
                    })}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </main>
  );
}