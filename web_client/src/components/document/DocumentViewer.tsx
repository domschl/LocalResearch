import { useQuery } from "@tanstack/react-query";
import { getDocument } from "@/lib/api";
import { Loader2 } from "lucide-react";

interface DocumentViewerProps {
  descriptor: string | null;
}

export function DocumentViewer({ descriptor }: DocumentViewerProps) {
  const { data: document, isLoading, error } = useQuery({
    queryKey: ["document", descriptor],
    queryFn: () => descriptor ? getDocument(descriptor) : null,
    enabled: !!descriptor,
  });

  if (!descriptor) {
    return (
      <div className="h-full flex items-center justify-center text-muted-foreground">
        Select a document to view
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-full flex items-center justify-center text-destructive">
        Error loading document
      </div>
    );
  }

  if (!document) return null;

  return (
    <div className="h-full flex flex-col overflow-hidden">
      <div className="p-6 border-b bg-card">
        <h1 className="text-2xl font-bold mb-2">{document.metadata.title}</h1>
        <div className="flex flex-wrap gap-4 text-sm text-muted-foreground">
          {document.metadata.authors?.length > 0 && (
            <div>Authors: {document.metadata.authors.join(", ")}</div>
          )}
          {document.metadata.creation_date && (
            <div>Created: {new Date(document.metadata.creation_date).toLocaleDateString()}</div>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-8 max-w-4xl mx-auto w-full">
        <div className="prose dark:prose-invert max-w-none">
          <pre className="whitespace-pre-wrap font-sans text-base">
            {document.content || "No content available"}
          </pre>
        </div>
      </div>
    </div>
  );
}
