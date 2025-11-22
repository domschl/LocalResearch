import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getVisualization3D } from "@/lib/api";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { Loader2 } from "lucide-react";

function PointCloud({ data, onSelect }: { data: any, onSelect: (index: number) => void }) {
  const { positions, colors } = useMemo(() => {
    if (!data) return { positions: new Float32Array(0), colors: new Float32Array(0) };

    const positions = new Float32Array(data.points.flat());
    const colors = new Float32Array(data.colors.flatMap((c: number[]) => [c[0]/255, c[1]/255, c[2]/255]));
    
    return { positions, colors };
  }, [data]);

  return (
    <points onClick={(e) => {
      e.stopPropagation();
      if (e.index !== undefined) {
        onSelect(e.index);
      }
    }}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={positions.length / 3}
          array={positions}
          itemSize={3}
          args={[positions, 3]}
        />
        <bufferAttribute
          attach="attributes-color"
          count={colors.length / 3}
          array={colors}
          itemSize={3}
          args={[colors, 3]}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.15}
        vertexColors
        sizeAttenuation
        transparent={false}
        alphaTest={0.5}
      />
    </points>
  );
}

export function ThreeDView() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["visualization3d"],
    queryFn: getVisualization3D,
    staleTime: Infinity, // Data is static per session usually
  });

  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <span className="ml-2">Loading 3D Data...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-full flex items-center justify-center text-destructive">
        Error loading 3D data: {(error as Error).message}
      </div>
    );
  }

  return (
    <div className="h-full w-full relative bg-black">
      <Canvas camera={{ position: [0, 0, 15], fov: 60 }}>
        <color attach="background" args={["#111"]} />
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} />
        <PointCloud data={data} onSelect={setSelectedIndex} />
        <OrbitControls makeDefault />
      </Canvas>
      
      {selectedIndex !== null && data && (
        <div className="absolute bottom-4 left-4 right-4 bg-card/90 p-4 rounded-lg border shadow-lg max-w-lg mx-auto backdrop-blur-sm">
          <h3 className="font-bold text-lg mb-1">{data.doc_ids[selectedIndex]}</h3>
          <p className="text-sm text-muted-foreground line-clamp-3">
            {data.texts[selectedIndex]}
          </p>
          <button 
            className="absolute top-2 right-2 text-muted-foreground hover:text-foreground"
            onClick={() => setSelectedIndex(null)}
          >
            ✕
          </button>
        </div>
      )}
      
      <div className="absolute top-4 left-4 bg-card/80 p-2 rounded text-xs text-muted-foreground pointer-events-none">
        {data?.points.length} points | Model: {data?.model_name}
      </div>
    </div>
  );
}
