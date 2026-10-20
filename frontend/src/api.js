import { API_BASE } from "./site.js";

export class ApiError extends Error {
  constructor(message, status, fields) {
    super(message);
    this.status = status;
    this.fields = fields || {};
  }
}

export async function api(path, { method = "GET", body, form, headers = {} } = {}) {
  const opts = { method, headers: { ...headers } };
  if (form) opts.body = form;
  else if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(`${API_BASE}/api${path}`, opts);
  } catch {
    throw new ApiError("The server could not be reached. Check that the backend is running.", 0);
  }
  let data = null;
  try {
    data = await res.json();
  } catch {
    /* non-JSON */
  }
  if (!res.ok) throw new ApiError((data && data.error) || `Request failed (${res.status})`, res.status, data && data.fields);
  return data;
}

export function userId() {
  try {
    let id = localStorage.getItem("uid");
    if (!id) {
      id = "u-" + Math.random().toString(36).slice(2, 12);
      localStorage.setItem("uid", id);
    }
    return id;
  } catch {
    return "anonymous";
  }
}
