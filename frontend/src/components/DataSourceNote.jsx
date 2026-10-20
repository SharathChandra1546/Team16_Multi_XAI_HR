import { Notice } from "./ui.jsx";

export default function DataSourceNote({ source }) {
  if (source === "synthetic_standin")
    return (
      <Notice role="note">
        Test data in use: the model was trained on a synthetic stand-in file for software testing. Numbers on this site
        are not results. Train on the IBM HR dataset before using it.
      </Notice>
    );
  if (source === "ibm_kaggle")
    return <p className="small muted">Trained on IBM's fictional HR Analytics Employee Attrition dataset (1,470 records).</p>;
  return null;
}
