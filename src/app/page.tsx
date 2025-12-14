"use client";

import { useEffect, useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { 
  Trash2, 
  Download, 
  PlusCircle, 
  Loader2, 
  Calendar, 
  Settings, 
  Globe,
  AlertCircle,
  CheckCircle2,
  X,
  Clock,
  Zap
} from "lucide-react";
import * as ort from "onnxruntime-web";

import DatePicker, { DateObject } from "react-multi-date-picker";
import persian from "react-date-object/calendars/persian";
import persian_fa from "react-date-object/locales/persian_fa";

type Task = {
  id?: string;
  name: string;
  priority: number;
  min_duration: number;
  max_duration: number;
  due_date: string;
};

const LS_TASKS_KEY = "planner_tasks_v2";
const LS_SCHEDULE_KEY = "planner_schedule_v1";
const LS_SETTINGS_KEY = "planner_settings_v1";
const MODEL_CACHE_KEY = "onnx_model_cache_v3";
const MODEL_URL = "/models/planner_single.onnx";

export default function Page() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [name, setName] = useState("");
  const [priority, setPriority] = useState(5);
  const [minD, setMinD] = useState(1.0);
  const [maxD, setMaxD] = useState(1.5);
  const [dueDate, setDueDate] = useState("");
  const [dailyHours, setDailyHours] = useState(8);
  const [weekendHours, setWeekendHours] = useState(4);
  const [rest, setRest] = useState(1);
  const [language, setLanguage] = useState<'en' | 'fa'>('fa');
  const [plan, setPlan] = useState<number[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [statusMsg, setStatusMsg] = useState<{type: 'error' | 'success', text: string} | null>(null);

  const sessionRef = useRef<ort.InferenceSession | null>(null);

  const showStatus = (type: 'error' | 'success', text: string) => {
    setStatusMsg({ type, text });
    setTimeout(() => setStatusMsg(null), 5000);
  };

  const formatNumber = (num: number | string) => {
    if (language !== 'fa') return String(num);
    const persianDigits = ["۰", "۱", "۲", "۳", "۴", "۵", "۶", "۷", "۸", "۹"];
    return String(num).replace(/[0-9]/g, (w) => persianDigits[+w]);
  };

  useEffect(() => {
    async function loadModel() {
      try {
        ort.env.wasm.numThreads = 1; 
        ort.env.wasm.simd = true;

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
        
        const session = await ort.InferenceSession.create(modelData, { 
          executionProviders: ["wasm"],
          graphOptimizationLevel: 'all'
        });
        sessionRef.current = session;
        setModelReady(true);
      } catch (err) {
        console.error("Failed to load ONNX model:", err);
        showStatus('error', "بارگذاری مدل هوش مصنوعی ناموفق بود. اتصال اینترنت خود را بررسی کنید.");
      }
    }
    loadModel();
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const savedLanguage = localStorage.getItem('preferred_language');
    if (savedLanguage === 'en' || savedLanguage === 'fa') setLanguage(savedLanguage);
    else setLanguage('en');
    
    const rawTasks = localStorage.getItem(LS_TASKS_KEY);
    if (rawTasks) {
      try {
        const parsed = JSON.parse(rawTasks);
        if (Array.isArray(parsed)) setTasks(parsed);
      } catch (err) { console.error(err); }
    }
    
    const rawSchedule = localStorage.getItem(LS_SCHEDULE_KEY);
    if (rawSchedule) {
      try {
        const parsed = JSON.parse(rawSchedule);
        setDailyHours(parsed.dailyHours ?? 8);
        setWeekendHours(parsed.weekendHours ?? 4);
      } catch (err) { console.error(err); }
    }
    
    const rawSettings = localStorage.getItem(LS_SETTINGS_KEY);
    if (rawSettings) {
      try {
        const parsed = JSON.parse(rawSettings);
        setRest(parsed.rest ?? 1.2);
      } catch (err) { console.error(err); }
    }
    
    setDueDate(new Date().toISOString().split("T")[0]);
  }, []);

  useEffect(() => {
    if (typeof window !== "undefined") localStorage.setItem(LS_TASKS_KEY, JSON.stringify(tasks));
  }, [tasks]);

  useEffect(() => {
    if (typeof window !== "undefined") localStorage.setItem(LS_SCHEDULE_KEY, JSON.stringify({ dailyHours, weekendHours }));
  }, [dailyHours, weekendHours]);

  useEffect(() => {
    if (typeof window !== "undefined") localStorage.setItem(LS_SETTINGS_KEY, JSON.stringify({ rest }));
  }, [rest]);

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

  function taskToVec(t: Task, currentTime = 0): number[] {
    const avgDur = (t.min_duration + t.max_duration) / 2;
    const availableHours = calculateAvailableHours(t.due_date);
    const slack = Math.max(0, availableHours - currentTime - avgDur);
    const durationSd = (t.max_duration - t.min_duration) / Math.sqrt(12);

    return [
      t.priority / 10.0,
      avgDur / 4.0,
      availableHours / 30.0,
      slack / 20.0,
      currentTime / 20.0,
      (t.priority * avgDur) / 40.0,
      durationSd / 2.0,
      durationSd / Math.max(avgDur, 0.1)
    ];
  }

  const isOverdue = (dateStr: string) => {
    const taskDate = new Date(`${dateStr}T23:59:59`);
    return taskDate < new Date();
  };

  async function getPlan(greedy = true) {
    if (!sessionRef.current) { 
      showStatus('error', "مدل هنوز بارگذاری نشده است. لطفا کمی صبر کنید."); 
      return; 
    }
    
    setLoading(true);
    setPlan(null);
    
    try {
      const validTaskIndices = tasks.map((_, i) => i).filter(i => !isOverdue(tasks[i].due_date));
      const apiTasks = validTaskIndices.map(i => tasks[i]);
      
      if (apiTasks.length === 0) {
        showStatus('error', "همه تکالیف مهلت‌شان گذشته است یا لیستی وجود ندارد.");
        setLoading(false);
        return;
      }

      const nTasks = apiTasks.length;
      const planIndices: number[] = [];
      const mask = new Array(nTasks).fill(true);
      let currentTime = 0.0;

      for (let step = 0; step < nTasks; step++) {
        const currentFeats: number[] = [];
        for (let i = 0; i < nTasks; i++) {
          currentFeats.push(...taskToVec(apiTasks[i], currentTime));
        }
        
        const featsTensor = new ort.Tensor('float32', Float32Array.from(currentFeats), [1, nTasks, 8]);
        const restTensor = new ort.Tensor('float32', Float32Array.from([rest]), [1]);
        const maskData = new Uint8Array(mask.map(m => m ? 1 : 0));
        const maskTensor = new ort.Tensor('bool', maskData, [1, nTasks]);

        const results = await sessionRef.current.run({
          task_features: featsTensor,
          rest_factor: restTensor,
          mask: maskTensor
        });

        const logits = results.logits.data as Float32Array;
        let selectedIdx = -1;
        
        const validLogits: {val: number, idx: number}[] = [];
        let maxL = -Infinity;
        
        for(let i=0; i<nTasks; i++) {
           if(mask[i]) {
             if(logits[i] > maxL) maxL = logits[i];
             validLogits.push({ val: logits[i], idx: i });
           }
        }
        
        const exps = validLogits.map(item => ({...item, exp: Math.exp(item.val - maxL)}));
        const sumExps = exps.reduce((acc, item) => acc + item.exp, 0);
        const probs = exps.map(item => ({...item, prob: item.exp / sumExps}));

        if (greedy) {
          const best = probs.reduce((prev, curr) => (prev.prob > curr.prob ? prev : curr), probs[0]);
          selectedIdx = best.idx;
        } else {
          const r = Math.random();
          let cum = 0;
          for (const p of probs) {
            cum += p.prob;
            if (r <= cum) {
              selectedIdx = p.idx;
              break;
            }
          }
          if (selectedIdx === -1 && probs.length > 0) selectedIdx = probs[probs.length - 1].idx;
        }

        if (selectedIdx !== -1) {
          planIndices.push(validTaskIndices[selectedIdx]);
          mask[selectedIdx] = false;
          
          const t = apiTasks[selectedIdx];
          const avgDur = (t.min_duration + t.max_duration) / 2;
          currentTime += avgDur * rest;
        }
      }

      setPlan(planIndices);
      showStatus('success', "برنامه با موفقیت ساخته شد!");

    } catch (err) {
      console.error("Planning error:", err);
      showStatus('error', "تولید برنامه با خطا مواجه شد. لطفا دوباره تلاش کنید.");
    }
    setLoading(false);
  }

  function addTask() {
    const newName = name?.trim() || `تکلیف شماره ${formatNumber(tasks.length + 1)}`;
    setTasks((prev) => [...prev, { 
      name: newName, 
      priority: priority || 5, 
      min_duration: minD || 1, 
      max_duration: maxD || 1.5, 
      due_date: dueDate 
    }]);
    setName("");
    showStatus('success', 'تکلیف اضافه شد');
  }

  function clearTasks() { 
    setTasks([]); 
    setPlan(null); 
  }

  async function clearCache() {
    try { 
      await caches.delete(MODEL_CACHE_KEY); 
      showStatus('success', "حافظه پنهان پاک شد. صفحه را رفرش کنید.");
      setTimeout(() => window.location.reload(), 1500);
    } catch (err) { console.error(err); }
  }

  function exportTasks() {
    const blob = new Blob([JSON.stringify(tasks, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "tasks.json"; a.click();
    URL.revokeObjectURL(url);
  }

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
    if (diff > 0 && diff < 7) return `تا ${formatNumber(diff)} روز دیگر`;
    return new DateObject(date).convert(persian, persian_fa).format("D MMMM");
  }

  function isoToJalaliDateObject(iso?: string) {
    try {
      if (!iso) return undefined;
      return new DateObject({ date: new Date(`${iso}T00:00:00`), calendar: persian, locale: persian_fa });
    } catch (e) { return undefined; }
  }

  const toggleLanguage = () => {
    const newLanguage = language === 'en' ? 'fa' : 'en';
    setLanguage(newLanguage);
    localStorage.setItem('preferred_language', newLanguage);
  };

  const translate = (persianText: string, englishText: string) => {
    return language === 'en' ? englishText : persianText;
  };

  const getPriorityColor = (priority: number) => {
    if (priority >= 8) return "bg-red-100 text-red-700 border-red-200";
    if (priority >= 5) return "bg-amber-100 text-amber-700 border-amber-200";
    return "bg-blue-100 text-blue-700 border-blue-200";
  };

  return (
    <main dir={language === 'en' ? 'ltr' : 'rtl'} className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50/30 to-gray-50 p-4 md:p-8">
      <link href="https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css" rel="stylesheet" type="text/css" />
      <style dangerouslySetInnerHTML={{ __html: `* { font-family: 'Vazirmatn', -apple-system, BlinkMacSystemFont, system-ui, sans-serif !important; }` }} />
      
      <AnimatePresence>
        {statusMsg && (
          <motion.div 
            initial={{ opacity: 0, y: -20 }} 
            animate={{ opacity: 1, y: 0 }} 
            exit={{ opacity: 0, y: -20 }}
            className="fixed top-4 left-1/2 transform -translate-x-1/2 z-50 px-6 py-3 rounded-full shadow-xl flex items-center gap-3 text-sm font-semibold backdrop-blur-xl"
            style={{
              background: statusMsg.type === 'error' 
                ? 'linear-gradient(135deg, rgba(239, 68, 68, 0.95), rgba(220, 38, 38, 0.95))' 
                : 'linear-gradient(135deg, rgba(34, 197, 94, 0.95), rgba(22, 163, 74, 0.95))',
              color: 'white'
            }}
          >
            {statusMsg.type === 'error' ? <AlertCircle size={18}/> : <CheckCircle2 size={18}/>}
            {statusMsg.text}
            <button onClick={() => setStatusMsg(null)} className="hover:opacity-80"><X size={16}/></button>
          </motion.div>
        )}
      </AnimatePresence>

      <motion.div 
        initial={{ opacity: 0 }} 
        animate={{ opacity: 1 }} 
        className="max-w-4xl mx-auto space-y-6"
      >
        
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <h1 className="text-3xl md:text-4xl font-black bg-gradient-to-l from-gray-900 via-gray-800 to-gray-900 bg-clip-text text-transparent">
              {translate('برنامه‌ریز هوشمند', 'Smart Planner')}
            </h1>
            <p className="text-sm text-gray-500 font-medium">{translate('مدیریت زمان با هوش مصنوعی', 'AI-Powered Task Scheduling')}</p>
          </div>
          <div className="flex items-center gap-2">
            <div className={`px-3 py-1.5 rounded-full text-xs font-bold flex items-center gap-2 transition-all ${modelReady ? 'bg-emerald-100 text-emerald-700' : 'bg-amber-100 text-amber-700'}`}>
              {modelReady ? <Zap size={14} className="animate-pulse" /> : <Loader2 size={14} className="animate-spin" />}
              {modelReady ? translate('مدل آماده است', 'Ready') : translate('در حال دریافت مدل', 'Loading')}
            </div>
            <Button variant="ghost" size="icon" onClick={toggleLanguage} className="rounded-full hover:bg-gray-100">
              <Globe size={18} />
            </Button>
          </div>
        </div>

        {/* Add Task Card */}
        <Card className="border-0 shadow-lg bg-white/80 backdrop-blur">
          <CardHeader className="border-b bg-gradient-to-r from-gray-50 to-white">
            <CardTitle className="text-lg font-bold flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-blue-500 flex items-center justify-center">
                <PlusCircle className="text-white" size={18} />
              </div>
              {translate('تکلیف جدید', 'New Task')}
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-6 space-y-5">
            <div className="space-y-2">
              <Label className="text-xs font-bold text-gray-600 uppercase tracking-wide">{translate('عنوان', 'Title')}</Label>
              <Input 
                value={name} 
                onChange={(e) => setName(e.target.value)} 
                placeholder={translate('مثال: مطالعه فصل سوم', 'e.g., Study Chapter 3')}
                className="h-12 bg-gray-50/50 border-gray-200 focus:border-blue-400 focus:ring-2 focus:ring-blue-100 transition-all"
                onKeyDown={(e) => e.key === 'Enter' && dueDate && addTask()}
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label className="text-xs font-bold text-gray-600 uppercase tracking-wide flex items-center gap-1.5">
                  <Calendar size={12} />
                  {translate('مهلت', 'Due Date')}
                </Label>
                <DatePicker 
                  calendar={persian} 
                  locale={persian_fa} 
                  value={isoToJalaliDateObject(dueDate)} 
                  onChange={(d: any) => { 
                    try { 
                      const iso = d.toDate().toISOString().split("T")[0]; 
                      setDueDate(iso); 
                    } catch (e) { console.error(e); } 
                  }} 
                  format="YYYY/MM/DD"
                  inputClass="w-full h-12 px-4 rounded-lg border border-gray-200 bg-gray-50/50 focus:border-blue-400 focus:ring-2 focus:ring-blue-100 outline-none transition-all text-sm font-medium"
                />
              </div>
              <div className="space-y-2">
                <Label className="text-xs font-bold text-gray-600 uppercase tracking-wide">{translate('اولویت', 'Priority')}</Label>
                <Input 
                  type="number" 
                  min={1} 
                  max={10} 
                  value={priority || ""} 
                  onChange={(e) => setPriority(Number(e.target.value))} 
                  className="h-12 text-center text-lg font-bold bg-gray-50/50 border-gray-200 focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label className="text-xs font-bold text-gray-600 uppercase tracking-wide">{translate('حداقل زمان (ساعت)', 'Min Hours')}</Label>
                <Input 
                  type="number" 
                  step="0.5" 
                  value={minD || ""} 
                  onChange={(e) => setMinD(Number(e.target.value))} 
                  className="h-12 text-center text-lg font-semibold bg-gray-50/50 border-gray-200 focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
                />
              </div>
              <div className="space-y-2">
                <Label className="text-xs font-bold text-gray-600 uppercase tracking-wide">{translate('حداکثر زمان (ساعت)', 'Max Hours')}</Label>
                <Input 
                  type="number" 
                  step="0.5" 
                  value={maxD || ""} 
                  onChange={(e) => setMaxD(Number(e.target.value))} 
                  className="h-12 text-center text-lg font-semibold bg-gray-50/50 border-gray-200 focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
                />
              </div>
            </div>

            {dueDate && (
              <div className="p-4 bg-gradient-to-r from-blue-50 to-cyan-50 border border-blue-100 rounded-xl">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-semibold text-gray-700 flex items-center gap-2">
                    <Clock size={16} className="text-blue-500" />
                    {translate('زمان در دسترس:', 'Available Time:')}
                  </span>
                  <span className="text-2xl font-black text-blue-600">{formatNumber(calculateAvailableHours(dueDate).toFixed(1))}</span>
                </div>
              </div>
            )}

            <div className="flex gap-3 pt-2">
              <Button 
                onClick={addTask} 
                disabled={!dueDate}
                className="flex-1 h-12 bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-700 hover:to-blue-600 text-white font-bold shadow-lg shadow-blue-200 disabled:opacity-50 disabled:shadow-none"
              >
                <PlusCircle size={20} className={language === 'en' ? 'mr-2' : 'ml-2'} />
                {translate('افزودن', 'Add Task')}
              </Button>
              <Button 
                variant="outline" 
                size="icon"
                onClick={() => setShowSettings(!showSettings)}
                className="h-12 w-12 border-gray-200 hover:bg-gray-50"
              >
                <Settings size={20} />
              </Button>
              {tasks.length > 0 && (
                <>
                  <Button 
                    variant="outline" 
                    size="icon"
                    onClick={exportTasks}
                    className="h-12 w-12 border-gray-200 hover:bg-gray-50"
                  >
                    <Download size={20} />
                  </Button>
                  <Button 
                    variant="outline" 
                    size="icon"
                    onClick={clearTasks}
                    className="h-12 w-12 border-red-200 hover:bg-red-50 text-red-500"
                  >
                    <Trash2 size={20} />
                  </Button>
                </>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Settings */}
        <AnimatePresence>
          {showSettings && (
            <motion.div 
              initial={{ opacity: 0, height: 0 }} 
              animate={{ opacity: 1, height: "auto" }} 
              exit={{ opacity: 0, height: 0 }}
            >
              <Card className="border-0 shadow-lg bg-gradient-to-br from-gray-50 to-white">
                <CardContent className="pt-6">
                  <div className="grid grid-cols-3 gap-4">
                    <div className="space-y-2">
                      <Label className="text-xs font-bold text-gray-600">{translate('روزهای عادی', 'Weekdays')}</Label>
                      <Input type="number" value={dailyHours} onChange={(e) => setDailyHours(Number(e.target.value))} className="h-10 text-center font-bold" />
                    </div>
                    <div className="space-y-2">
                      <Label className="text-xs font-bold text-gray-600">{translate('آخر هفته', 'Weekend')}</Label>
                      <Input type="number" value={weekendHours} onChange={(e) => setWeekendHours(Number(e.target.value))} className="h-10 text-center font-bold" />
                    </div>
                    <div className="space-y-2">
                      <Label className="text-xs font-bold text-gray-600">{translate('ضریب استراحت', 'Rest Factor')}</Label>
                      <Input type="number" step="0.1" value={rest} onChange={(e) => setRest(Number(e.target.value))} className="h-10 text-center font-bold" />
                    </div>
                  </div>
                  <div className="mt-4 flex justify-end">
                    <Button variant="link" size="sm" onClick={clearCache} className="text-xs text-gray-400 hover:text-red-500">
                      {translate('پاک کردن کش', 'Clear Cache')}
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Tasks List */}
        {tasks.length > 0 && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-black text-gray-800 flex items-center gap-2">
                {plan ? translate('برنامه پیشنهادی', 'Optimized Schedule') : translate('تکالیف', 'Tasks')}
                <span className="px-2.5 py-1 bg-gray-900 text-white text-sm rounded-full font-bold">{formatNumber(tasks.length)}</span>
              </h2>
              {plan && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setPlan(null)}
                  className="text-gray-500 hover:text-gray-700 font-semibold"
                >
                  {translate('بازگشت به حالت اولیه', 'Reset Order')}
                </Button>
              )}
            </div>

            <div className={`space-y-0 rounded-2xl overflow-hidden border-2 ${plan ? 'bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 border-gray-700' : 'border-gray-100'}`}>
              {(plan ? plan.map(idx => ({ task: tasks[idx], originalIdx: idx })) : tasks.map((task, idx) => ({ task, originalIdx: idx }))).map((item, displayPos) => {
                if (!item.task) return null;
                const task = item.task;
                const overdue = isOverdue(task.due_date);
                const avgDuration = ((task.min_duration + task.max_duration) / 2).toFixed(1);
                const isPlanned = !!plan;
                
                return (
                  <motion.div
                    key={item.originalIdx}
                    layout
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: displayPos * 0.05 }}
                    className={`group relative p-5 transition-all ${
                      isPlanned
                        ? 'border-b border-gray-700/50 last:border-0 hover:bg-gray-800/50'
                        : `border-b-2 last:border-0 hover:shadow-lg ${
                            overdue 
                              ? 'bg-red-50/50 border-red-200' 
                              : 'bg-white border-gray-100 hover:border-blue-200'
                          }`
                    }`}
                  >
                    <div className="flex items-start gap-4">
                      <div className={`flex-shrink-0 w-12 h-12 rounded-xl flex items-center justify-center font-black text-lg shadow-lg ${
                        isPlanned
                          ? 'bg-gradient-to-br from-blue-500 to-blue-600 text-white shadow-blue-500/30'
                          : overdue 
                            ? 'bg-red-500 text-white' 
                            : 'bg-gradient-to-br from-gray-700 to-gray-900 text-white'
                      }`}>
                        {overdue && !isPlanned ? <AlertCircle size={20} /> : formatNumber(displayPos + 1)}
                      </div>
                      
                      <div className="flex-1 min-w-0">
                        <div className="flex items-start justify-between gap-3 mb-2">
                          <h3 className={`font-bold text-lg leading-tight ${
                            isPlanned 
                              ? 'text-white' 
                              : overdue 
                                ? 'line-through text-gray-400' 
                                : 'text-gray-900'
                          }`}>
                            {task.name}
                          </h3>
                          {!isPlanned && (
                            <button
                              onClick={() => setTasks(prev => prev.filter((_, idx) => idx !== item.originalIdx))}
                              className="flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center text-gray-300 hover:bg-red-100 hover:text-red-500 transition-all"
                            >
                              <X size={16} />
                            </button>
                          )}
                        </div>
                        
                        <div className="flex flex-wrap gap-2">
                          {isPlanned ? (
                            <>
                              <span className="inline-flex items-center gap-1 text-xs text-gray-400">
                                <Zap size={12} className="text-blue-400" />
                                {translate('اولویت', 'Priority')} <span className="text-blue-300 font-bold">{formatNumber(task.priority)}</span>
                              </span>
                              <span className="inline-flex items-center gap-1 text-xs text-gray-400">
                                <Clock size={12} className="text-emerald-400" />
                                {translate('تخمین', 'Est')} <span className="text-emerald-300 font-bold">{formatNumber(avgDuration)}h</span>
                              </span>
                            </>
                          ) : (
                            <>
                              <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-bold border ${getPriorityColor(task.priority)}`}>
                                <Zap size={12} />
                                {formatNumber(task.priority)}
                              </span>
                              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-bold bg-gray-100 text-gray-700 border border-gray-200">
                                <Clock size={12} />
                                {formatNumber(avgDuration)}h
                              </span>
                              <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-bold ${
                                overdue 
                                  ? 'bg-red-100 text-red-700 border border-red-200' 
                                  : 'bg-blue-100 text-blue-700 border border-blue-200'
                              }`}>
                                <Calendar size={12} />
                                {formatDueDate(task.due_date)}
                              </span>
                            </>
                          )}
                        </div>
                      </div>

                      {isPlanned && (
                        <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-gray-700/50 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                          <CheckCircle2 size={16} className="text-emerald-400" />
                        </div>
                      )}
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </div>
        )}

        {/* Generate Plan Buttons */}
        {tasks.length > 0 && (
          <div className="sticky bottom-6 z-10 flex justify-center">
            <motion.div 
              initial={{ y: 100 }}
              animate={{ y: 0 }}
              className="bg-white/95 backdrop-blur-xl p-3 rounded-2xl shadow-2xl border border-gray-200 flex gap-3"
            >
              <Button
                onClick={() => getPlan(true)}
                disabled={loading || !modelReady}
                className="h-14 px-8 bg-gradient-to-r from-gray-900 to-gray-700 hover:from-gray-800 hover:to-gray-600 text-white font-bold rounded-xl shadow-lg disabled:opacity-50"
              >
                {loading ? (
                  <>
                    <Loader2 className="animate-spin" size={20} />
                    <span className={language === 'en' ? 'ml-2' : 'mr-2'}>{translate('در حال پردازش...', 'Processing...')}</span>
                  </>
                ) : (
                  <>
                    <Zap size={20} className={language === 'en' ? 'mr-2' : 'ml-2'} />
                    {translate('تولید برنامه', 'Generate Plan')}
                  </>
                )}
              </Button>
              <Button
                onClick={() => getPlan(false)}
                disabled={loading || !modelReady}
                variant="outline"
                className="h-14 px-6 rounded-xl border-2 border-gray-200 hover:bg-gray-50 font-bold disabled:opacity-50"
              >
                {translate('تصادفی', 'Random')}
              </Button>
            </motion.div>
          </div>
        )}



        {/* Empty State */}
        {tasks.length === 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center py-20"
          >
            <div className="w-20 h-20 bg-gradient-to-br from-gray-100 to-gray-200 rounded-3xl flex items-center justify-center mx-auto mb-6 shadow-inner">
              <Calendar className="text-gray-400" size={40} />
            </div>
            <h3 className="text-xl font-bold text-gray-800 mb-2">{translate('لیست خالی است', 'No Tasks Yet')}</h3>
            <p className="text-gray-500 text-sm">{translate('برای شروع، یک تکلیف اضافه کنید', 'Add your first task to get started')}</p>
          </motion.div>
        )}

      </motion.div>
    </main>
  );
}
