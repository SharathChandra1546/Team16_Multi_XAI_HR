import { Notice } from "./ui.jsx";

export default function DataSourceNote({ source, ui }) {
  const notes = ui?.dataset_notes || {};
  if (source === "synthetic_standin")
    return (
      <Notice role="note">
        Test data in use: the model was trained on {notes.synthetic_standin || "a synthetic stand-in file for software testing."}{" "}
        Numbers on this site are not results.
      </Notice>
    );
  return notes[source] ? <p className="small muted">{notes[source]}</p> : null;
}
