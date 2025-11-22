import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getTimeline, type TimelineEvent } from "@/lib/api";
import { Timeline, type TimelineOptions } from "vis-timeline/standalone";
import { DataSet } from "vis-data";
import "vis-timeline/styles/vis-timeline-graph2d.css";
import { v4 as uuidv4 } from 'uuid';
import { IndraTime } from "@/lib/indra_time";

export function TimelineView() {
  const containerRef = useRef<HTMLDivElement>(null);
  const timelineRef = useRef<Timeline | null>(null);
  const [selectedEvent, setSelectedEvent] = useState<TimelineEvent | null>(null);
  const [keywords, setKeywords] = useState<string>("");
  const [startTime, setStartTime] = useState<string>("");
  const [endTime, setEndTime] = useState<string>("");
  const [domains, setDomains] = useState<string>("");
  
  const [debouncedKeywords, setDebouncedKeywords] = useState<string>("");
  const [debouncedStartTime, setDebouncedStartTime] = useState<string>("");
  const [debouncedEndTime, setDebouncedEndTime] = useState<string>("");
  const [debouncedDomains, setDebouncedDomains] = useState<string>("");

  // Debounce all filters
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedKeywords(keywords);
      setDebouncedStartTime(startTime);
      setDebouncedEndTime(endTime);
      setDebouncedDomains(domains);
    }, 500);
    return () => clearTimeout(timer);
  }, [keywords, startTime, endTime, domains]);

  const timeString = (debouncedStartTime || debouncedEndTime) 
    ? `${debouncedStartTime} - ${debouncedEndTime}` 
    : undefined;

  const { data, error } = useQuery({
    queryKey: ["timeline", debouncedKeywords, timeString, debouncedDomains],
    queryFn: () => getTimeline(debouncedKeywords, timeString, debouncedDomains),
    enabled: debouncedKeywords.length > 0 || (!!timeString && timeString.length > 3) || debouncedDomains.length > 0,
  });

  // Mapping constants
  // Map 1 year to 100 milliseconds to fit 14 billion years into JS Date range (+/- 270,000 years)
  // 14 * 10^9 years * 0.1s/year = 1.4 * 10^9 seconds = 44 years. This fits easily.
  // Julian Date J2000 (Jan 1, 2000) = 2451545.0
  const J2000 = 2451545.0;
  const MS_PER_YEAR = 100; 
  const DAYS_PER_YEAR = 365.25;
  const SCALE = MS_PER_YEAR / DAYS_PER_YEAR; // ms per Julian Day
  const OFFSET = new Date("2000-01-01").getTime();

  const jdToTimeline = (jd: number) => {
    return (jd - J2000) * SCALE + OFFSET;
  };

  const timelineToJd = (time: number) => {
    return (time - OFFSET) / SCALE + J2000;
  };

  useEffect(() => {
    if (containerRef.current && !timelineRef.current) {
        // Initialize empty timeline
        const items = new DataSet([]);
        const options: TimelineOptions = {
            height: "100%",
            width: "100%",
            verticalScroll: true,
            zoomKey: "ctrlKey",
            orientation: { axis: "top", item: "top" },
            showCurrentTime: false,
            format: {
              minorLabels: (date: Date | number, _scale: string, _step: number) => {
                const time = date instanceof Date ? date.getTime() : (typeof date === 'number' ? date : new Date(date).getTime());
                const jd = timelineToJd(time);
                return IndraTime.julianToStringTime(jd);
              },
              majorLabels: (date: Date | number, _scale: string, _step: number) => {
                const time = date instanceof Date ? date.getTime() : (typeof date === 'number' ? date : new Date(date).getTime());
                const jd = timelineToJd(time);
                return IndraTime.julianToStringTime(jd);
              }
            }
          };
    
          timelineRef.current = new Timeline(containerRef.current, items, options);
    
          timelineRef.current.on("select", (properties) => {
            if (properties.items && properties.items.length > 0) {
               // Selection logic temporarily removed to simplify
               setSelectedEvent(null);
            } else {
              setSelectedEvent(null);
            }
          });
    }

    if (timelineRef.current && data) {
        // Update data
        const parsedItems = data.map((item) => {
            const jds = IndraTime.stringTimeToJulian(item.date);
            if (jds.length === 0) return null;
            
            const start = jdToTimeline(jds[0]);
            if (isNaN(start)) return null;
    
            let end = undefined;
            if (jds.length > 1) {
              end = jdToTimeline(jds[1]);
              if (isNaN(end)) return null;
            }
            
            return {
              id: uuidv4(),
              content: item.event.length > 50 ? item.event.substring(0, 50) + "..." : item.event,
              start: start, 
              end: end,
              title: item.event,
              fullItem: item,
              type: end ? 'range' : 'point'
            };
          }).filter(i => i !== null);
          
          timelineRef.current.setItems(parsedItems); // Pass array directly
          timelineRef.current.fit();
    } else if (timelineRef.current && !data && debouncedKeywords.length === 0) {
        // Clear items if no keywords
        timelineRef.current.setItems([]);
    }

  }, [data, debouncedKeywords]);

  // Cleanup on unmount
  useEffect(() => {
      return () => {
          if (timelineRef.current) {
              timelineRef.current.destroy();
              timelineRef.current = null;
          }
      }
  }, []);
  
  // Let's rewrite the init logic to be cleaner.
  

  if (error) {
    return (
      <div className="h-full flex items-center justify-center text-destructive">
        Error loading timeline: {(error as Error).message}
      </div>
    );
  }

  return (
    <div className="h-full w-full flex flex-col relative overflow-hidden">
      <div className="p-4 border-b bg-card flex flex-col gap-4 z-10">
        <div className="flex gap-4 items-center">
            <input 
                type="text" 
                placeholder="Keywords (e.g. 'italy')" 
                className="border rounded px-3 py-2 w-full bg-background"
                value={keywords}
                onChange={(e) => setKeywords(e.target.value)}
            />
            <input 
                type="text" 
                placeholder="Domains (e.g. 'history')" 
                className="border rounded px-3 py-2 w-full bg-background"
                value={domains}
                onChange={(e) => setDomains(e.target.value)}
            />
        </div>
        <div className="flex gap-4 items-center">
            <input 
                type="text" 
                placeholder="Start Time (e.g. '1000')" 
                className="border rounded px-3 py-2 w-full bg-background"
                value={startTime}
                onChange={(e) => setStartTime(e.target.value)}
            />
            <span className="text-muted-foreground">-</span>
            <input 
                type="text" 
                placeholder="End Time (e.g. '2000')" 
                className="border rounded px-3 py-2 w-full bg-background"
                value={endTime}
                onChange={(e) => setEndTime(e.target.value)}
            />
            <span className="text-sm text-muted-foreground whitespace-nowrap min-w-[100px] text-right">
                {data?.length || 0} events
            </span>
        </div>
      </div>
      
      <div ref={containerRef} className="flex-1 w-full relative">
        {(!data || data.length === 0) && debouncedKeywords.length === 0 && !timeString && debouncedDomains.length === 0 && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-20">
                <div className="bg-card/80 p-6 rounded-lg shadow-lg text-center backdrop-blur-sm border">
                    <h3 className="text-lg font-semibold mb-2">Enter search criteria</h3>
                    <p className="text-muted-foreground">
                        Filter by keywords, time range, or domains to explore the timeline.
                    </p>
                </div>
            </div>
        )}
      </div>
      
      {selectedEvent && (
        <div className="absolute bottom-4 left-4 right-4 bg-card/90 p-4 rounded-lg border shadow-lg max-w-3xl mx-auto backdrop-blur-sm z-10">
          <h3 className="font-bold text-lg mb-1">{selectedEvent.date}</h3>
          <p className="text-sm text-muted-foreground">
            {selectedEvent.event}
          </p>
          <button 
            className="absolute top-2 right-2 text-muted-foreground hover:text-foreground"
            onClick={() => {
              if (timelineRef.current) {
                timelineRef.current.setSelection([]);
              }
              setSelectedEvent(null);
            }}
          >
            ✕
          </button>
        </div>
      )}
    </div>
  );
}
