import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import { Sidebar } from "./Sidebar";
import { Outlet } from "react-router-dom";

export function AppLayout() {
  return (
    <div className="h-screen w-screen bg-background text-foreground overflow-hidden flex flex-col">
      <header className="h-12 border-b flex items-center px-4 shrink-0 bg-card">
        <h1 className="font-bold text-lg">LocalResearch</h1>
      </header>
      <div className="flex-1 overflow-hidden">
        <PanelGroup direction="horizontal">
          <Panel defaultSize={20} minSize={15} maxSize={30} className="border-r">
            <Sidebar />
          </Panel>
          <PanelResizeHandle className="w-1 bg-border hover:bg-primary/50 transition-colors cursor-col-resize" />
          <Panel defaultSize={80}>
             <Outlet />
          </Panel>
        </PanelGroup>
      </div>
    </div>
  );
}
