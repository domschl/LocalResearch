import { useState } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AppLayout } from "./components/layout/AppLayout";
import { SearchPane } from "./components/search/SearchPane";
import { DocumentViewer } from "./components/document/DocumentViewer";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";

import { ThreeDView } from "./components/visualization/ThreeDView";
import { TimelineView } from "./components/visualization/TimelineView";

function SearchPage() {
  const [selectedDescriptor, setSelectedDescriptor] = useState<string | null>(null);

  return (
    <PanelGroup direction="horizontal">
      <Panel defaultSize={30} minSize={20} maxSize={50} className="border-r">
        <SearchPane onSelectDocument={setSelectedDescriptor} />
      </Panel>
      <PanelResizeHandle className="w-1 bg-border hover:bg-primary/50 transition-colors cursor-col-resize" />
      <Panel defaultSize={70}>
        <DocumentViewer descriptor={selectedDescriptor} />
      </Panel>
    </PanelGroup>
  );
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppLayout />}>
          <Route path="/" element={<SearchPage />} />
          <Route path="/3d" element={<ThreeDView />} />
          <Route path="/timeline" element={<TimelineView />} />
          <Route path="/settings" element={<div className="p-4">Settings (Placeholder)</div>} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
