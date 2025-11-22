import axios from 'axios';

export const api = axios.create({
  baseURL: 'http://localhost:8000',
});

export interface MetadataEntry {
  uuid: string;
  title: string;
  authors: string[];
  description: string;
  tags: string[];
  creation_date: string;
  // Add other fields as needed
}

export interface SearchResult {
  id: number;
  score: number;
  descriptor: string;
  text: string | null;
  metadata: MetadataEntry | null;
}

export interface DocumentResponse {
  descriptor: string;
  metadata: MetadataEntry;
  path: string;
  content: string | null;
}

export interface VisualizationData {
  points: number[][];
  colors: number[][];
  texts: string[];
  doc_ids: string[];
  sizes: number[];
  model_name?: string;
  reduction_method?: string;
  num_points_visualized?: number;
  num_points_available_before_sampling?: number;
}

export const searchDocuments = async (query: string): Promise<SearchResult[]> => {
  const response = await api.get<SearchResult[]>('/search', { params: { q: query } });
  return response.data;
};

export const keywordSearchDocuments = async (query: string): Promise<SearchResult[]> => {
  const response = await api.get<SearchResult[]>('/ksearch', { params: { q: query } });
  return response.data;
};

export const getDocument = async (descriptor: string): Promise<DocumentResponse> => {
  // Descriptor might contain special characters, so we encode it
  // However, FastAPI path parameters might handle it differently. 
  // Let's try direct path first, but encoded.
  const encodedDescriptor = encodeURIComponent(descriptor);
  const response = await api.get<DocumentResponse>(`/document/${encodedDescriptor}`);
  return response.data;
};

export interface TimelineEvent {
  date: string;
  event: string;
}

export const getVisualization3D = async (): Promise<VisualizationData> => {
  const response = await api.get<VisualizationData>('/visualization/3d');
  return response.data;
};

export const getTimeline = async (keywords?: string): Promise<TimelineEvent[]> => {
  const params = keywords ? { keywords } : {};
  const response = await api.get<TimelineEvent[]>('/timeline', { params });
  return response.data;
};
