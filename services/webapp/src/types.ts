export type A2UIMessage = Record<string, any>;

export type UIEvent = {
  type: string;
  status?: string;
  delta?: string;
  name?: string;
  step?: number;
  max_steps?: number;
  content?: string;
  raw_assistant_message?: Record<string, unknown>;
  arguments?: Record<string, unknown>;
  tool_calls?: Array<Record<string, unknown>>;
  result?: unknown;
  deduplicated?: boolean;
  answer?: string;
  has_ui?: boolean;
  state?: Record<string, unknown> | null;
  message?: A2UIMessage;
  error?: string;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  uiMessages?: A2UIMessage[];
  status?: string;
  tools?: string[];
  trace?: UIEvent[];
  error?: string;
};
