import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getTimeline, type TimelineEvent } from "@/lib/api";
import { Loader2 } from "lucide-react";
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
  const [debouncedKeywords, setDebouncedKeywords] = useState<string>("");

  // Debounce keywords
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedKeywords(keywords);
    }, 500);
    return () => clearTimeout(timer);
  }, [keywords]);

  const { data, isLoading, error } = useQuery({
    queryKey: ["timeline", debouncedKeywords],
    queryFn: () => getTimeline(debouncedKeywords),
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
    if (containerRef.current && data && !timelineRef.current) {
      // console.log("Timeline data received:", data.length);
      
      const parsedItems = data.map((item) => {
        const jds = IndraTime.stringTimeToJulian(item.date);
        if (jds.length === 0) {
            return null;
        }
        
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
      // console.log("Parsed items count:", parsedItems.length);

      const items = new DataSet(parsedItems);

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
          const id = properties.items[0];
          const item = items.get(id);
          if (item) {
             setSelectedEvent((item as any).fullItem);
          }
        } else {
          setSelectedEvent(null);
        }
      });
    } else if (timelineRef.current && data) {
        // Update data if timeline exists
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
          
          timelineRef.current.setItems(new DataSet(parsedItems));
          timelineRef.current.fit();
    }

    return () => {
      // Cleanup only on unmount, not on data change if we update in place
      // But here we destroy and recreate if container is missing or first load
      // Actually, let's just destroy if we unmount.
    };
  }, [data]);

  // Cleanup on unmount
  useEffect(() => {
      return () => {
          if (timelineRef.current) {
              timelineRef.current.destroy();
              timelineRef.current = null;
          }
      }
  }, []);

  if (isLoading && !data) {
    return (
      <div className="h-full flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <span className="ml-2">Loading Timeline...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-full flex items-center justify-center text-destructive">
        Error loading timeline: {(error as Error).message}
      </div>
    );
  }

  return (
    <div className="h-full w-full flex flex-col relative overflow-hidden">
      <div className="p-4 border-b bg-card flex gap-4 items-center z-10">
        <input 
            type="text" 
            placeholder="Filter by keywords (e.g. 'italy')" 
            className="border rounded px-3 py-2 w-full max-w-md bg-background"
            value={keywords}
            onChange={(e) => setKeywords(e.target.value)}
        />
        <span className="text-sm text-muted-foreground">
            {data?.length || 0} events found
        </span>
      </div>
      
      <div ref={containerRef} className="flex-1 w-full" />
      
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
