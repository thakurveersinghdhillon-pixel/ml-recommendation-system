export interface ModelHealthStatus {
  content_based: string;
  collaborative: string;
  hybrid: string;
}

export interface HealthResponse {
  status: string;
  service: string;
  database?: string;
  models: ModelHealthStatus;
}

export interface ApiErrorDetail {
  code: string;
  message: string;
}

export interface ApiErrorResponse {
  error: ApiErrorDetail;
}

export class CustomApiError extends Error {
  code: string;
  statusCode: number;

  constructor(code: string, message: string, statusCode: number = 400) {
    super(message);
    this.name = "CustomApiError";
    this.code = code;
    this.statusCode = statusCode;
  }
}
