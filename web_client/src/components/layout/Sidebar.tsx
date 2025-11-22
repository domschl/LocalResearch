import { Search, Box, Calendar, Settings } from "lucide-react";
import { Link, useLocation } from "react-router-dom";
import { cn } from "@/lib/utils";

const navItems = [
  { icon: Search, label: "Search", path: "/" },
  { icon: Box, label: "3D View", path: "/3d" },
  { icon: Calendar, label: "Timeline", path: "/timeline" },
  { icon: Settings, label: "Settings", path: "/settings" },
];

export function Sidebar() {
  const location = useLocation();

  return (
    <nav className="flex flex-col gap-2 p-2 h-full bg-muted/10">
      {navItems.map((item) => (
        <Link
          key={item.path}
          to={item.path}
          className={cn(
            "flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground",
            location.pathname === item.path ? "bg-accent text-accent-foreground" : "text-muted-foreground"
          )}
        >
          <item.icon className="h-4 w-4" />
          {item.label}
        </Link>
      ))}
    </nav>
  );
}
