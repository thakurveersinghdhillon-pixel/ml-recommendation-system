export interface User {
  id: number;
  username: string;
  email: string;
  created_at: string;
  updated_at: string;
}

export interface UserCreate {
  username: string;
  email: string;
}

export interface UserUpdate {
  username?: string;
  email?: string;
}
