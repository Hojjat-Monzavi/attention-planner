"use client";

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Trash2, Download, PlusCircle, Loader2, Calendar, Clock, Settings } from "lucide-react";

type Task = {
  name: string;
  priority: number;
  min_duration: number;
  max_duration: number;
  due_date: string;
  available_hours: number;
};

type DailySchedule = {
  [key: string]: number;
};

const LS_TASKS_KEY = "planner_tasks_v2";
const LS_SCHEDULE_KEY = "planner_schedule_v1";

export default function Page() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [name, setName] = useState("");
  const [priority, setPriority] = useState(5);
  const [minD, setMinD] = useState(1.0);
  const [maxD, setMaxD] = useState(1.5);
  const [dueDate, setDueDate] = useState("");
  const [dueTime, setDueTime] = useState("17:00");
  const [dailyHours, setDailyHours] = useState(8);
  const [weekendHours, setWeekendHours] = useState(4);
  const [rest, setRest] = useState(1.2);
  const [plan, setPlan] = useState<number[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

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

  function calculateAvailableHours(dueDate: string, dueTime: string): number {
    const now = new Date();
    const deadline = new Date(`${dueDate}T${dueTime}`);
    
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

  function addTask() {
    const newName = name?.trim() || `Task ${tasks.length + 1}`;
    const availableHours = calculateAvailableHours(dueDate, dueTime);
    
    setTasks((prev) => [
      ...prev,
      { 
        name: newName, 
        priority, 
        min_duration: Number(minD), 
        max_duration: Number(maxD), 
        due_date: dueDate,
        available_hours: availableHours
      },
    ]);
    setName("");
    
    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1);
    setDueDate(tomorrow.toISOString().split('T')[0]);
  }

  function clearTasks() {
    setTasks([]);
    setPlan(null);
  }

  async function getPlan(greedy = true) {
    setLoading(true);
    setPlan(null);

    try {
      const apiTasks = tasks.map(({ name: _n, due_date: _d, available_hours, ...rest }) => ({
        ...rest,
        due_time: available_hours
      }));

      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tasks: apiTasks, rest_factor: rest, greedy }),
      });

      const text = await res.text();
      let planArr: number[] | null = null;

      try {
        const parsed = JSON.parse(text);
        if (Array.isArray(parsed)) planArr = parsed.map((v) => Number(v));
        else if (Array.isArray(parsed?.plan)) planArr = parsed.plan.map((v: any) => Number(v));
        else if (Array.isArray(parsed?.data?.plan)) planArr = parsed.data.plan.map((v: any) => Number(v));
      } catch (e) {
        const lines = text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
        const indices: number[] = [];
        const indexRegex = /#\s*(\d+)/g;
        for (const line of lines) {
          let m: RegExpExecArray | null;
          while ((m = indexRegex.exec(line)) !== null) {
            indices.push(Number(m[1]));
          }
        }
        if (indices.length) planArr = indices;
      }

      if (!planArr) {
        const nums = Array.from(text.matchAll(/\b(\d+)\b/g)).map((m) => Number(m[1]));
        if (nums.length) planArr = nums;
      }

      if (!planArr) throw new Error("Unable to parse plan from response");

      const sanitized = planArr.map((n) => Math.max(0, Math.min(tasks.length - 1, Number(n))));
      setPlan(sanitized);
    } catch (err) {
      alert("Failed to generate plan. Please try again.");
    }

    setLoading(false);
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

                <div>
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

                <div>
                  <Label className="text-sm font-medium flex items-center gap-2">
                    <Clock size={14} /> Due Time
                  </Label>
                  <Input 
                    type="time" 
                    value={dueTime} 
                    onChange={(e) => setDueTime(e.target.value)}
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
                    value={priority}
                    onChange={(e) => setPriority(Number(e.target.value))}
                    className="mt-1"
                  />
                  <p className="text-xs text-slate-500 mt-1">1-10 scale</p>
                </div>

                <div>
                  <Label className="text-sm font-medium">Min Duration</Label>
                  <Input 
                    type="number" 
                    step="0.5" 
                    value={minD} 
                    onChange={(e) => setMinD(Number(e.target.value))}
                    className="mt-1"
                  />
                  <p className="text-xs text-slate-500 mt-1">hours</p>
                </div>

                <div>
                  <Label className="text-sm font-medium">Max Duration</Label>
                  <Input 
                    type="number" 
                    step="0.5" 
                    value={maxD} 
                    onChange={(e) => setMaxD(Number(e.target.value))}
                    className="mt-1"
                  />
                  <p className="text-xs text-slate-500 mt-1">hours</p>
                </div>
              </div>

              {dueDate && (
                <div className="p-3 bg-indigo-50 rounded-lg border border-indigo-200">
                  <p className="text-sm text-indigo-900">
                    <strong>Available time:</strong> {calculateAvailableHours(dueDate, dueTime).toFixed(1)} hours 
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
                        value={dailyHours} 
                        onChange={(e) => setDailyHours(Number(e.target.value))}
                        className="mt-1"
                      />
                      <p className="text-xs text-slate-500 mt-1">Mon-Fri available hours</p>
                    </div>

                    <div>
                      <Label className="text-sm font-medium">Weekend Hours</Label>
                      <Input 
                        type="number" 
                        step="0.5" 
                        value={weekendHours} 
                        onChange={(e) => setWeekendHours(Number(e.target.value))}
                        className="mt-1"
                      />
                      <p className="text-xs text-slate-500 mt-1">Sat-Sun available hours</p>
                    </div>

                    <div>
                      <Label className="text-sm font-medium">Rest Factor</Label>
                      <Input 
                        type="number" 
                        step="0.1" 
                        value={rest} 
                        onChange={(e) => setRest(Number(e.target.value))}
                        className="mt-1"
                      />
                      <p className="text-xs text-slate-500 mt-1">Planning buffer multiplier</p>
                    </div>
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
          <div className="flex justify-center gap-4">
            <Button 
              onClick={() => getPlan(true)} 
              disabled={loading}
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
              disabled={loading}
              size="lg"
            >
              Sample Plan
            </Button>
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