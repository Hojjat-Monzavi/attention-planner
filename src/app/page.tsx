"use client";

import { useEffect, useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Trash2, Download, PlusCircle, Loader2, Calendar, Clock, Settings, AlertTriangle } from "lucide-react";
import * as ort from "onnxruntime-web";

import DatePicker, { DateObject } from "react-multi-date-picker";
import persian from "react-date-object/calendars/persian";
import persian_fa from "react-date-object/locales/persian_fa";

type Task = {
  name: string;
  priority: number;
  min_duration: number;
  max_duration: number;
  due_date: string; // ISO YYYY-MM-DD
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

  const toPersianDigits = (num: number | string) => {
    const persianDigits = ["۰", "۱", "۲", "۳", "۴", "۵", "۶", "۷", "۸", "۹"];
    return String(num).replace(/[0-9]/g, (w) => persianDigits[+w]);
  };

  useEffect(() => {
    async function loadModel() {
      try {
        const cache = await caches.open(MODEL_CACHE_KEY);
        let modelData: ArrayBuffer;
        const cachedResponse = await cache.match(MODEL_URL);
        if (cachedResponse) {
          modelData = await cachedResponse.arrayBuffer();
        } else {
          const response = await fetch(MODEL_URL);
          if (!response.ok) throw new Error("Failed to fetch model");
          await cache.put(MODEL_URL, response.clone());
          modelData = await response.arrayBuffer();
        }
        const session = await ort.InferenceSession.create(modelData, { executionProviders: ["wasm"] });
        sessionRef.current = session;
        setModelReady(true);
      } catch (err) {
        console.error("Failed to load ONNX model:", err);
        setModelError("بارگذاری مدل هوش مصنوعی ناموفق بود. اتصال اینترنت خود را بررسی کنید و صفحه را بازنشانی کنید.");
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
            name: t.name || `تکلیف ${i + 1}`,
            priority: t.priority ?? 5,
            min_duration: t.min_duration ?? 1,
            max_duration: t.max_duration ?? 1.5,
            due_date: t.due_date || new Date().toISOString().split("T")[0],
            available_hours: t.available_hours ?? 0,
          })));
        }
      } catch (err) { console.error("Failed to parse tasks", err); }
    }
    const rawSchedule = localStorage.getItem(LS_SCHEDULE_KEY);
    if (rawSchedule) {
      try {
        const parsed = JSON.parse(rawSchedule);
        setDailyHours(parsed.dailyHours ?? 8);
        setWeekendHours(parsed.weekendHours ?? 4);
      } catch (err) { console.error("Failed to parse schedule", err); }
    }
    const today = new Date();
    setDueDate(today.toISOString().split("T")[0]);
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try { localStorage.setItem(LS_TASKS_KEY, JSON.stringify(tasks)); } 
    catch (e) { console.error("Failed saving tasks", e); }
  }, [tasks]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try { localStorage.setItem(LS_SCHEDULE_KEY, JSON.stringify({ dailyHours, weekendHours })); } 
    catch (e) { console.error("Failed saving schedule", e); }
  }, [dailyHours, weekendHours]);

  // FIX 1: Replaced with a more robust calculation function.
  function calculateAvailableHours(dueDateISO: string): number {
    const now = new Date();
    const deadline = new Date(`${dueDateISO}T17:00:00`);

    if (deadline <= now) return 0;

    let totalHours = 0;
    const startOfToday = new Date();
    startOfToday.setHours(0, 0, 0, 0);

    const startOfDeadlineDay = new Date(deadline);
    startOfDeadlineDay.setHours(0, 0, 0, 0);

    if (startOfToday.getTime() === startOfDeadlineDay.getTime()) {
      const workdayStart = 9;
      const nowHour = now.getHours() + now.getMinutes() / 60;
      const deadlineHour = deadline.getHours() + deadline.getMinutes() / 60;
      const isWeekend = now.getDay() === 4 || now.getDay() === 5;
      const dayHours = isWeekend ? weekendHours : dailyHours;
      const workdayEnd = workdayStart + dayHours;
      const start = Math.max(workdayStart, nowHour);
      const end = Math.min(workdayEnd, deadlineHour);
      totalHours = Math.max(0, end - start);
    } else {
      const workdayStart = 9;
      const nowHour = now.getHours() + now.getMinutes() / 60;
      const isTodayWeekend = now.getDay() === 4 || now.getDay() === 5;
      const todayTotalHours = isTodayWeekend ? weekendHours : dailyHours;
      const todayWorkdayEnd = workdayStart + todayTotalHours;
      totalHours += Math.max(0, todayWorkdayEnd - nowHour);

      let loopDay = new Date(startOfToday);
      loopDay.setDate(loopDay.getDate() + 1);

      while (loopDay.getTime() < startOfDeadlineDay.getTime()) {
        const isLoopDayWeekend = loopDay.getDay() === 4 || loopDay.getDay() === 5;
        totalHours += isLoopDayWeekend ? weekendHours : dailyHours;
        loopDay.setDate(loopDay.getDate() + 1);
      }
      const isDeadlineDayWeekend = deadline.getDay() === 4 || deadline.getDay() === 5;
      const deadlineDayTotalHours = isDeadlineDayWeekend ? weekendHours : dailyHours;
      const deadlineHour = deadline.getHours() + deadline.getMinutes() / 60;
      totalHours += Math.max(0, Math.min(deadlineDayTotalHours, deadlineHour - workdayStart));
    }
    return Math.max(0, Math.round(totalHours * 10) / 10);
  }

  function taskToVec(t: { priority: number; min_duration: number; max_duration: number; available_hours: number }, currentTime = 0): number[] {
    const avgDur = (t.min_duration + t.max_duration) / 2;
    const slack = Math.max(0, t.available_hours - currentTime - avgDur);
    return [t.priority/10, avgDur/4, t.available_hours/30, slack/20, currentTime/20, (t.priority * avgDur)/40];
  }

  async function getPlan(greedy = true) {
    if (!sessionRef.current) { alert("مدل هنوز بارگذاری نشده است. لطفا کمی صبر کنید."); return; }
    setLoading(true);
    setPlan(null);
    try {
      const apiTasks = tasks.map(({ available_hours, ...rest }) => ({ ...rest, due_time: available_hours }));
      const planIndices: number[] = [];
      const remaining = apiTasks.map((_, i) => i);
      let currentTime = 0;
      while (remaining.length > 0) {
        const feats = remaining.map((i) => taskToVec({ priority: apiTasks[i].priority, min_duration: apiTasks[i].min_duration, max_duration: apiTasks[i].max_duration, available_hours: apiTasks[i].due_time }, currentTime));
        const tensor = new ort.Tensor("float32", new Float32Array(feats.flat()), [feats.length, 6]);
        const maskTensor = new ort.Tensor("float32", new Float32Array(feats.length).fill(1.0), [feats.length]);
        const restTensor = new ort.Tensor("float32", Float32Array.from([rest]), [1]);
        const results = await sessionRef.current.run({ task_features: tensor, rest_factor: restTensor, mask: maskTensor });
        const logits = results.logits.data as Float32Array;
        const maxLogit = Math.max(...Array.from(logits));
        const exps = Array.from(logits).map((v) => Math.exp(v - maxLogit));
        const probs = exps.map((e) => e / exps.reduce((a, b) => a + b, 0));
        let idxSelected = 0;
        if (greedy) { idxSelected = probs.indexOf(Math.max(...probs)); } 
        else { const r = Math.random(); let cum = 0; idxSelected = probs.findIndex((p) => (cum += p) >= r); if (idxSelected < 0) idxSelected = probs.length - 1; }
        const selected = remaining[idxSelected];
        planIndices.push(selected);
        const t = apiTasks[selected];
        currentTime += ((t.min_duration + t.max_duration) / 2) * rest;
        remaining.splice(idxSelected, 1);
      }
      setPlan(planIndices);
    } catch (err) { console.error("Planning error:", err); alert("تولید برنامه با خطا مواجه شد. دوباره تلاش کنید."); }
    setLoading(false);
  }

  function addTask() {
    const newName = name?.trim() || `تکلیف شماره ${toPersianDigits(tasks.length + 1)}`;
    const availableHours = calculateAvailableHours(dueDate);
    setTasks((prev) => [...prev, { name: newName, priority: priority || 5, min_duration: minD || 1, max_duration: maxD || 1.5, due_date: dueDate, available_hours: availableHours }]);
  }

  function clearTasks() { setTasks([]); setPlan(null); }

  async function clearCache() {
    try { await caches.delete(MODEL_CACHE_KEY); alert("حافظه پنهان مدل پاک شد! لطفا صفحه را دوباره بارگذاری کنید."); } 
    catch (err) { console.error("Failed to clear cache:", err); }
  }

  function exportTasks() {
    const blob = new Blob([JSON.stringify(tasks, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "tasks.json"; a.click();
    URL.revokeObjectURL(url);
  }

  // FIX 1 (cont.): Use more robust date comparison to fix today/tomorrow display bug.
  function formatDueDate(dateStr: string): string {
    const date = new Date(`${dateStr}T00:00:00`);
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);

    if (date.getTime() === today.getTime()) return "امروز";
    if (date.getTime() === tomorrow.getTime()) return "فردا";

    if (date < today) return "مهلت گذشته";

    const diff = Math.ceil((date.getTime() - today.getTime()) / (1000 * 60 * 60 * 24));
    if (diff > 0 && diff < 7) return `تا ${toPersianDigits(diff)} روز دیگر`;
    return new DateObject(date).convert(persian, persian_fa).format("D MMMM");
  }
  
  // FIX 3: Helper function to check if a task is overdue.
  const isOverdue = (dateStr: string) => {
    const taskDate = new Date(`${dateStr}T23:59:59`);
    return taskDate < new Date();
  };

  function isoToJalaliDateObject(iso?: string) {
    try {
      if (!iso) return undefined;
      return new DateObject({ date: new Date(`${iso}T00:00:00`), calendar: persian, locale: persian_fa });
    } catch (e) { return undefined; }
  }

  return (
    <main dir="rtl" className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50 p-4 md:p-8" style={{ fontFamily: "Vazir, Vazirmatn, sans-serif" }}>
      <style dangerouslySetInnerHTML={{ __html: `@import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazir-font@v30.1.0/dist/font-face.css'); html, body { direction: rtl; }` }} />
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} className="max-w-4xl mx-auto space-y-6">
        <div className="text-center space-y-2 mb-8">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent"> برنامه‌ریز هوشمند  </h1>
          <p className="text-slate-600">برنامه ریزی و اولیت بندی با کمک شبکه های عصبی مصنوعی و مکانیسم Attention</p>
          {modelReady ? ( <div className="inline-flex items-center gap-2 px-3 py-1 bg-green-100 text-green-700 rounded-full text-xs font-medium"> <div className="w-2 h-2 bg-green-500 rounded-full"></div> مدل هوش مصنوعی آماده است </div> ) : 
                       ( <div className="inline-flex items-center gap-2 px-3 py-1 bg-amber-100 text-amber-700 rounded-full text-xs font-medium"> <Loader2 className="w-3 h-3 animate-spin" /> در حال بارگذاری مدل... </div> )}
        </div>
        <Card className="shadow-lg border-indigo-100">
          <CardHeader className="bg-gradient-to-r from-indigo-50 to-purple-50"> <CardTitle className="text-xl font-semibold flex items-center gap-2"> <PlusCircle className="text-indigo-600" size={20} /> افزودن تکلیف جدید </CardTitle> </CardHeader>
          <CardContent className="pt-6">
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="md:col-span-2">
                  <Label className="text-sm font-medium">عنوان تکلیف</Label>
                  <Input value={name} onChange={(e) => setName(e.target.value)} placeholder={`مثال: مطالعه فیزیولوژی`} className="mt-1" />
                </div>
                <div className="md:col-span-2">
                  <Label className="text-sm font-medium flex items-center gap-2"> <Calendar size={14} /> مهلت انجام </Label>
                  <DatePicker calendar={persian} locale={persian_fa} value={isoToJalaliDateObject(dueDate)} onChange={(d: any) => { try { const jsd: Date = d.toDate(); const iso = jsd.toISOString().split("T")[0]; setDueDate(iso); } catch (e) { console.error(e); } }} format="YYYY/MM/DD" inputClass="w-full mt-1" />
                </div>
              </div>
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <Label className="text-sm font-medium">اولویت</Label>
                  <Input type="number" min={1} max={10} value={priority || ""} onChange={(e) => setPriority(e.target.value === "" ? 0 : Number(e.target.value))} placeholder="۵" className="mt-1" />
                  <p className="text-xs text-slate-500 mt-1">عددی بین ۱ تا ۱۰</p>
                </div>
                <div>
                  <Label className="text-sm font-medium">حداقل زمان</Label>
                  <Input type="number" step="0.5" min={0} value={minD || ""} onChange={(e) => setMinD(e.target.value === "" ? 0 : Number(e.target.value))} placeholder="۱" className="mt-1" />
                  <p className="text-xs text-slate-500 mt-1">(ساعت)</p>
                </div>
                <div>
                  <Label className="text-sm font-medium">حداکثر زمان</Label>
                  <Input type="number" step="0.5" min={0} value={maxD || ""} onChange={(e) => setMaxD(e.target.value === "" ? 0 : Number(e.target.value))} placeholder="۱.۵" className="mt-1" />
                  <p className="text-xs text-slate-500 mt-1">(ساعت)</p>
                </div>
              </div>
              {dueDate && ( <div className="p-3 bg-indigo-50 rounded-lg border border-indigo-200"> <p className="text-sm text-indigo-900"> <strong>زمان در دسترس شما:</strong> {toPersianDigits(calculateAvailableHours(dueDate).toFixed(1))} ساعت <span className="text-indigo-600"> · {formatDueDate(dueDate)}</span> </p> </div> )}
              <div className="flex flex-wrap gap-3 pt-2">
                <Button onClick={addTask} className="bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700" disabled={!dueDate}> <PlusCircle size={16} className="ml-2" /> افزودن تکلیف </Button>
                <Button variant="outline" onClick={() => setShowSettings(!showSettings)}> <Settings size={16} className="ml-2" /> تنظیمات </Button>
                <Button variant="outline" onClick={exportTasks}> <Download size={16} className="ml-2" /> دریافت خروجی </Button>
                {tasks.length > 0 && (<Button variant="destructive" onClick={clearTasks}>پاک‌ کردن همه</Button>)}
              </div>
            </div>
          </CardContent>
        </Card>
        <AnimatePresence> {showSettings && ( <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }}> <Card className="shadow-md border-slate-200"> <CardHeader> <CardTitle className="text-lg font-semibold">تنظیمات برنامه‌ریزی</CardTitle> </CardHeader> <CardContent> <div className="grid grid-cols-1 md:grid-cols-3 gap-4"> <div> <Label className="text-sm font-medium">ساعات کاری (روزهای عادی)</Label> <Input type="number" step="0.5" min={0} value={dailyHours || ""} onChange={(e) => setDailyHours(e.target.value === "" ? 0 : Number(e.target.value))} placeholder="۸" className="mt-1" /> <p className="text-xs text-slate-500 mt-1">شنبه تا چهارشنبه</p> </div> <div> <Label className="text-sm font-medium">ساعات کاری (آخر هفته)</Label> <Input type="number" step="0.5" min={0} value={weekendHours || ""} onChange={(e) => setWeekendHours(e.target.value === "" ? 0 : Number(e.target.value))} placeholder="۴" className="mt-1" /> <p className="text-xs text-slate-500 mt-1">پنجشنبه و جمعه</p> </div> <div> <Label className="text-sm font-medium">ضریب استراحت</Label> <Input type="number" step="0.1" min={1} value={rest || ""} onChange={(e) => setRest(e.target.value === "" ? 0 : Number(e.target.value))} placeholder="۱.۲" className="mt-1" /> <p className="text-xs text-slate-500 mt-1">برای محاسبه زمان استراحت بین کارها</p> </div> </div> <div className="mt-4 pt-4 border-t"> <Button variant="outline" size="sm" onClick={clearCache}> پاک‌سازی حافظه پنهان مدل </Button> <p className="text-xs text-slate-500 mt-2">برای دریافت آخرین نسخه مدل، حافظه پنهان را پاک کنید.</p> </div> </CardContent> </Card> </motion.div> )} </AnimatePresence>
        <Card className="shadow-md">
          <CardHeader> <CardTitle className="text-lg font-semibold">لیست تکالیف ({toPersianDigits(tasks.length)})</CardTitle> </CardHeader>
          <CardContent>
            {tasks.length === 0 ? ( <div className="text-center py-12"> <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-4"> <Calendar className="text-slate-400" size={32} /> </div> <p className="text-slate-500">هنوز تکلیفی ثبت نکرده‌اید. اولین تکلیف خود را اضافه کنید!</p> </div> ) : 
            ( <div className="space-y-3"> {tasks.map((t, i) => {
                const overdue = isOverdue(t.due_date);
                return (
                  <motion.div key={i} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2, delay: i * 0.05 }}
                    className={`group p-4 rounded-lg border bg-white hover:shadow-md transition-all ${ overdue ? "border-red-200 bg-red-50/70 text-slate-500" : "border-slate-200 hover:border-indigo-300" }`}>
                    <div className="flex items-start gap-4">
                      <div className={`w-12 h-12 flex-shrink-0 flex items-center justify-center rounded-lg text-white font-bold text-lg ${ overdue ? "bg-gradient-to-br from-red-400 to-rose-500" : "bg-gradient-to-br from-indigo-500 to-purple-600" }`}>
                        {overdue ? <AlertTriangle size={24}/> : toPersianDigits(i + 1)}
                      </div>
                      <div className="flex-1 min-w-0">
                        <h3 className={`font-semibold mb-2 ${ overdue ? "line-through text-red-800" : "text-slate-900" }`}> {t.name} </h3>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
                          <div> <span className="font-medium">اولویت:</span> {toPersianDigits(t.priority)}/۱۰ </div>
                          <div> <span className="font-medium">زمان:</span> {toPersianDigits(t.min_duration)}-{toPersianDigits(t.max_duration)} ساعت </div>
                          <div className={`${overdue ? 'font-bold text-red-600' : ''}`}> <span className="font-medium">مهلت:</span> {formatDueDate(t.due_date)} </div>
                          <div> <span className="font-medium">زمان موجود:</span> {toPersianDigits(t.available_hours.toFixed(1))} ساعت </div>
                        </div>
                      </div>
                      <Button size="icon" variant="ghost" className="opacity-0 group-hover:opacity-100 transition-opacity" onClick={() => setTasks((prev) => prev.filter((_, idx) => idx !== i))}> <Trash2 size={16} className="text-red-500" /> </Button>
                    </div>
                  </motion.div>
                );
              })} </div>
            )}
          </CardContent>
        </Card>
        {tasks.length > 0 && ( <div className="flex flex-col items-center gap-3"> {!modelReady && ( <div className="text-sm text-amber-600 flex items-center gap-2"> <Loader2 className="w-4 h-4 animate-spin" /> در انتظار بارگذاری مدل... </div> )} <div className="flex gap-4"> <Button onClick={() => getPlan(true)} disabled={loading || !modelReady} size="lg" className="bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 px-8"> {loading ? ( <> <Loader2 className="animate-spin ml-2" size={18} /> در حال ساخت برنامه... </> ) : ( "ساخت برنامه بهینه" )} </Button> <Button variant="outline" onClick={() => getPlan(false)} disabled={loading || !modelReady} size="lg"> ساخت برنامه نیمه تصادفی </Button> </div> </div> )}
        <AnimatePresence> {plan && ( <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.95 }}> <Card className="shadow-lg border-green-200 bg-gradient-to-br from-green-50 to-emerald-50"> <CardHeader> <CardTitle className="text-xl font-semibold text-green-900 flex items-center gap-2"> <div className="w-8 h-8 bg-green-600 rounded-full flex items-center justify-center text-white text-sm">✓</div> برنامه پیشنهادی شما </CardTitle> </CardHeader> <CardContent> <div className="space-y-3"> {plan.map((idx, pos) => { const t = tasks[idx]; if (!t) return null; const avg = ((t.min_duration + t.max_duration) / 2).toFixed(1); return ( <motion.div key={pos} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.3, delay: pos * 0.08 }} className="flex items-center gap-4 p-4 rounded-lg bg-white border border-green-200 shadow-sm"> <div className="w-10 h-10 rounded-full bg-gradient-to-br from-green-500 to-emerald-600 text-white flex items-center justify-center font-bold"> {toPersianDigits(pos + 1)} </div> <div className="flex-1 min-w-0"> <h4 className="font-semibold text-slate-900">{t.name}</h4> <div className="flex flex-wrap gap-3 mt-1 text-xs text-slate-600"> <span>حدود {toPersianDigits(avg)} ساعت</span> <span>·</span> <span>اولویت {toPersianDigits(t.priority)}</span> <span>·</span> <span>مهلت {formatDueDate(t.due_date)}</span> </div> </div> </motion.div> ); })} </div> </CardContent> </Card> </motion.div> )} </AnimatePresence>
      </motion.div>
      <footer className="mt-12 py-6 text-center text-slate-500 text-sm border-t border-slate-200">
        توسعه داده شده توسط حجت منزوی
      </footer>
    </main>
  );
}