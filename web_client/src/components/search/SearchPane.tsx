import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { searchDocuments, keywordSearchDocuments } from "@/lib/api";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Search as SearchIcon, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface SearchPaneProps {
  onSelectDocument: (descriptor: string) => void;
}

export function SearchPane({ onSelectDocument }: SearchPaneProps) {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<"semantic" | "keyword">("semantic");

  const { data: results, isLoading, refetch } = useQuery({
    queryKey: ["search", mode, query],
    queryFn: () => mode === "semantic" ? searchDocuments(query) : keywordSearchDocuments(query),
    enabled: false, // Don't search automatically on type for now
  });

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      refetch();
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="p-4 border-b space-y-4">
        <form onSubmit={handleSearch} className="flex gap-2">
          <Input 
            value={query} 
            onChange={(e) => setQuery(e.target.value)} 
            placeholder="Search documents..." 
            className="flex-1"
          />
          <Button type="submit" disabled={isLoading}>
            {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <SearchIcon className="h-4 w-4" />}
          </Button>
        </form>
        <div className="flex gap-2 text-sm">
          <button 
            onClick={() => setMode("semantic")}
            className={cn("px-3 py-1 rounded-full border transition-colors", mode === "semantic" ? "bg-primary text-primary-foreground border-primary" : "hover:bg-muted")}
          >
            Semantic
          </button>
          <button 
            onClick={() => setMode("keyword")}
            className={cn("px-3 py-1 rounded-full border transition-colors", mode === "keyword" ? "bg-primary text-primary-foreground border-primary" : "hover:bg-muted")}
          >
            Keyword
          </button>
        </div>
      </div>
      
      <div className="flex-1 overflow-auto p-4 space-y-4">
        {results?.map((result) => (
          <div 
            key={result.id} 
            onClick={() => onSelectDocument(result.descriptor)}
            className="p-4 rounded-lg border hover:border-primary cursor-pointer transition-colors bg-card text-card-foreground shadow-sm"
          >
            <div className="flex justify-between items-start mb-2">
              <h3 className="font-semibold line-clamp-1">{result.metadata?.title || "Untitled"}</h3>
              <span className="text-xs text-muted-foreground bg-muted px-2 py-1 rounded">{result.score.toFixed(2)}</span>
            </div>
            <p className="text-sm text-muted-foreground line-clamp-3 mb-2">
              {result.text || result.metadata?.description || "No preview available"}
            </p>
            <div className="flex flex-wrap gap-1">
              {result.metadata?.tags?.slice(0, 3).map(tag => (
                <span key={tag} className="text-xs bg-secondary text-secondary-foreground px-2 py-0.5 rounded-full">
                  {tag}
                </span>
              ))}
            </div>
          </div>
        ))}
        {results?.length === 0 && (
          <div className="text-center text-muted-foreground py-8">
            No results found
          </div>
        )}
      </div>
    </div>
  );
}
