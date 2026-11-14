
import { useEffect, useState } from "react";
import axios from "axios";
import "./App.css";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

function App() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [loadingModels, setLoadingModels] = useState(true);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [analytics, setAnalytics] = useState(null);
  const [loadingAnalytics, setLoadingAnalytics] = useState(true);
  const [analyticsError, setAnalyticsError] = useState("");

  // Model registration
  const [form, setForm] = useState({
    model_name: "",
    description: "",
    framework: "scikit-learn",
    version: "1.0.0",
    input_schema: JSON.stringify(
      {
        features: [
          "sepal_length",
          "sepal_width",
          "petal_length",
          "petal_width",
        ],
      },
      null,
      2
    ),
    output_description: "Predicted Iris flower class",
  });

  const [modelFile, setModelFile] = useState(null);

  // CSV Dataset Training
  const [datasetFile, setDatasetFile] = useState(null);
  const [datasetModelName, setDatasetModelName] = useState("");
  const [targetColumn, setTargetColumn] = useState("");
  const [trainingTask, setTrainingTask] = useState("classification");
  const [training, setTraining] = useState(false);
  const [trainingMessage, setTrainingMessage] = useState("");
  const [trainingError, setTrainingError] = useState("");
  const [trainingMetrics, setTrainingMetrics] = useState(null); 
  // CSV Inspection
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [inspectingDataset, setInspectingDataset] = useState(false);
  const [inspectionError, setInspectionError] = useState("");

  // Prediction form
  const [selectedModelId, setSelectedModelId] = useState("");
  const [predictionSchema, setPredictionSchema] = useState([]);
  const [predictionValues, setPredictionValues] = useState({});
  const [loadingSchema, setLoadingSchema] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [predictionError, setPredictionError] = useState("");

  // Prediction history
  const [history, setHistory] = useState([]);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [historyError, setHistoryError] = useState("");

  const validModels = models.filter(
    (model) => model.validation_status === "VALID"
  );

  // ----------------------------------------
  // FETCH REGISTERED MODELS
  // ----------------------------------------

  async function fetchModels() {
    setLoadingModels(true);

    try {
     const response = await axios.get(API_URL + "/api/models/");

      const data = response.data;
      const modelList = Array.isArray(data)
        ? data
        : data.models || data.items || [];

      setModels(modelList);
      setError("");

      const validList = modelList.filter(
        (model) => model.validation_status === "VALID"
      );

      setSelectedModelId((currentId) => {
        const stillValid = validList.some(
          (model) => String(model.id) === String(currentId)
        );

        if (stillValid) return currentId;

        return validList.length ? String(validList[0].id) : "";
      });
    } catch {
      setError(
        "Could not load models. Make sure the FastAPI backend is running."
      );
    } finally {
      setLoadingModels(false);
    }
  }

  // ----------------------------------------
  // FETCH PREDICTION HISTORY
  // ----------------------------------------

  async function fetchHistory() {
    setLoadingHistory(true);
    setHistoryError("");

    try {
      const response = await axios.get(
        `${API_URL}/api/models/predictions/history`,
        {
          params: { limit: 100 },
        }
      );

      const data = response.data;

      setHistory(Array.isArray(data) ? data : data.history || []);
    } catch (err) {
      setHistoryError(
        err.response?.data?.detail ||
          "Could not load prediction history. Check the backend."
      );
    } finally {
      setLoadingHistory(false);
    }
  }

  // ----------------------------------------
  // FETCH ANALYTICS
  // ----------------------------------------

  async function fetchAnalytics() {
    setLoadingAnalytics(true);
    setAnalyticsError("");

    try {
      const response = await axios.get(
        `${API_URL}/api/models/analytics/summary`
      );

      setAnalytics(response.data);
    } catch (err) {
      setAnalyticsError(
        err.response?.data?.detail ||
          "Could not load prediction analytics."
      );
    } finally {
      setLoadingAnalytics(false);
    }
  }

  // ----------------------------------------
  // INITIAL DATA LOADING
  // ----------------------------------------

  useEffect(() => {
    fetchModels();
    fetchHistory();
    fetchAnalytics();
  }, []);

  // ----------------------------------------
  // LOAD SELECTED MODEL'S INPUT SCHEMA
  // ----------------------------------------

  useEffect(() => {
    if (!selectedModelId) {
      setPredictionSchema([]);
      setPredictionValues({});
      setPredictionResult(null);
      setPredictionError("");
      setLoadingSchema(false);
      return;
    }

    let cancelled = false;

    async function fetchModelDetails() {
      setLoadingSchema(true);
      setPredictionError("");
      setPredictionResult(null);

      try {
        const response = await axios.get(
          `${API_URL}/api/models/${selectedModelId}`
        );

        const model = response.data;
        let schema = model.input_schema;

        if (typeof schema === "string") {
          schema = JSON.parse(schema);
        }

        const rawFeatures = schema?.features;

        // Support both legacy string schemas and new CSV schemas.
        const features = Array.isArray(rawFeatures)
          ? rawFeatures.map((feature) =>
              typeof feature === "string"
                ? { name: feature, type: "numeric" }
                : feature
            )
          : [];

        if (
          features.length === 0 ||
          features.some(
            (feature) =>
              !feature ||
              typeof feature.name !== "string" ||
              !feature.name.trim()
          )
        ) {
          throw new Error(
            "This model does not have a valid input feature schema."
          );
        }

        if (!cancelled) {
          setPredictionSchema(features);

          const modelName = String(
            model.model_name || ""
          ).toLowerCase();

          const isIrisModel = modelName.includes("iris");

          const irisDefaults = {
            sepal_length: 5.1,
            sepal_width: 3.5,
            petal_length: 1.4,
            petal_width: 0.2,
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2,
          };

          const initialValues = {};

          features.forEach((featureDefinition) => {
            const feature = featureDefinition.name;

            if (
              isIrisModel &&
              irisDefaults[feature] !== undefined
            ) {
              initialValues[feature] = String(
                irisDefaults[feature]
              );
            } else {
              initialValues[feature] = "";
            }
          });

          setPredictionValues(initialValues);
        }
      } catch (err) {
        if (!cancelled) {
          setPredictionSchema([]);
          setPredictionValues({});

          setPredictionError(
            err.response?.data?.detail ||
              err.message ||
              "Could not load model input features."
          );
        }
      } finally {
        if (!cancelled) {
          setLoadingSchema(false);
        }
      }
    }

    fetchModelDetails();

    return () => {
      cancelled = true;
    };
  }, [selectedModelId]);

  // ----------------------------------------
  // MODEL REGISTRATION
  // ----------------------------------------

  function handleChange(event) {
    const { name, value } = event.target;

    setForm((previous) => ({
      ...previous,
      [name]: value,
    }));
  }

  async function handleSubmit(event) {
    event.preventDefault();

    setMessage("");
    setError("");

    if (!modelFile) {
      setError("Please select a model file.");
      return;
    }

    let parsedSchema;

    try {
      parsedSchema = JSON.parse(form.input_schema);

      if (
        !parsedSchema ||
        !Array.isArray(parsedSchema.features) ||
        parsedSchema.features.length === 0
      ) {
        setError(
          'Input schema must contain a non-empty "features" array.'
        );
        return;
      }

      if (
        !parsedSchema.features.every(
          (feature) =>
            typeof feature === "string" && feature.trim()
        )
      ) {
        setError(
          "Every feature must have a non-empty string name."
        );
        return;
      }
    } catch {
      setError("Input schema must be valid JSON.");
      return;
    }

    const formData = new FormData();

    formData.append("model_name", form.model_name);
    formData.append("description", form.description);
    formData.append("framework", form.framework);
    formData.append("version", form.version);
    formData.append("input_schema", JSON.stringify(parsedSchema));
    formData.append("output_description", form.output_description);
    formData.append("model_file", modelFile);

    setLoading(true);

    try {
      const response = await axios.post(
        `${API_URL}/api/models/register`,
        formData
      );

      setMessage(
        response.data.validation_message ||
          response.data.message ||
          "Model registration completed."
      );

      setForm((previous) => ({
        ...previous,
        model_name: "",
        description: "",
      }));

      setModelFile(null);

      const fileInput = document.getElementById("model-file");

      if (fileInput) fileInput.value = "";

      await Promise.all([fetchModels(), fetchAnalytics()]);
    } catch (err) {
      const detail = err.response?.data?.detail;

      setError(
        typeof detail === "string"
          ? detail
          : Array.isArray(detail)
            ? detail.map((item) => item.msg).join(", ")
            : "Model registration failed. Check the backend terminal."
      );
    } finally {
      setLoading(false);
    }
  }

  // ----------------------------------------
  // INSPECT CSV DATASET
  // ----------------------------------------

  async function handleDatasetInspection(file) {
    setDatasetInfo(null);
    setTargetColumn("");
    setTrainingTask("classification");
    setInspectionError("");
    setTrainingError("");
    setTrainingMessage("");

    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".csv")) {
      setInspectionError("Please select a valid CSV file.");
      return;
    }

    const formData = new FormData();
    formData.append("dataset_file", file);

    setInspectingDataset(true);

    try {
      const response = await axios.post(
        `${API_URL}/api/datasets/inspect`,
        formData
      );

      const data = response.data;

      setDatasetInfo(data);

      if (data.suggested_target) {
        setTargetColumn(data.suggested_target);
      }

      if (
        data.suggested_task === "classification" ||
        data.suggested_task === "regression"
      ) {
        setTrainingTask(data.suggested_task);
      }
    } catch (err) {
      const detail = err.response?.data?.detail;

      setInspectionError(
        typeof detail === "string"
          ? detail
          : "Could not inspect CSV dataset. Check the backend."
      );
    } finally {
      setInspectingDataset(false);
    }
  }

  // ----------------------------------------
  // TRAIN MODEL FROM CSV DATASET
  // ----------------------------------------

  async function handleDatasetTraining(event) {
    event.preventDefault();

    setTrainingMessage("");
    setTrainingError("");

    if (!datasetFile) {
      setTrainingError("Please select a CSV dataset.");
      return;
    }

    if (!datasetInfo) {
      setTrainingError("Please wait for CSV inspection to finish.");
      return;
    }

    if (!datasetModelName.trim() || !targetColumn.trim()) {
      setTrainingError("Enter a model name and select a target column.");
      return;
    }

    const formData = new FormData();

    formData.append("dataset_file", datasetFile);
    formData.append("model_name", datasetModelName.trim());
    formData.append("target_column", targetColumn.trim());
    formData.append("task", trainingTask);

    setTraining(true);

    try {
      const response = await axios.post(
        `${API_URL}/api/datasets/train`,
        formData
      );

      setTrainingMessage(
        response.data.message ||
          `Model "${datasetModelName}" trained and registered successfully.`
      );

      setTrainingMetrics(response.data.model?.metrics || null);

      setDatasetFile(null);
      setDatasetInfo(null);
      setDatasetModelName("");
      setTargetColumn("");
      setTrainingTask("classification");

      const fileInput = document.getElementById("dataset-file");

      if (fileInput) fileInput.value = "";

      await Promise.all([
        fetchModels(),
        fetchAnalytics(),
        fetchHistory(),
      ]);
    } catch (err) {
      const detail = err.response?.data?.detail;

      setTrainingError(
        typeof detail === "string"
          ? detail
          : Array.isArray(detail)
            ? detail.map((item) => item.msg).join(", ")
            : "Dataset training failed. Check the backend terminal."
      );
    } finally {
      setTraining(false);
    }
  }

  // ----------------------------------------
  // PREDICTION INPUT CHANGES
  // ----------------------------------------

  function handlePredictionValueChange(feature, value) {
    setPredictionValues((previous) => ({
      ...previous,
      [feature]: value,
    }));

    setPredictionResult(null);
    setPredictionError("");
  }

  // ----------------------------------------
  // RUN PREDICTION
  // ----------------------------------------

  async function handlePredict(event) {
    event.preventDefault();

    setPredictionError("");
    setPredictionResult(null);

    if (!selectedModelId) {
      setPredictionError("Please select a validated model.");
      return;
    }

    if (predictionSchema.length === 0) {
      setPredictionError(
        "The selected model has no input feature schema."
      );
      return;
    }

    const features = {};

    for (const featureDefinition of predictionSchema) {
      const feature = featureDefinition.name;
      const rawValue = predictionValues[feature];

      if (
        rawValue === undefined ||
        String(rawValue).trim() === ""
      ) {
        setPredictionError(
          `Please enter a value for "${feature}".`
        );
        return;
      }

      const value = String(rawValue).trim();

      if (featureDefinition.type === "numeric") {
        const numericValue = Number(value);

        if (!Number.isFinite(numericValue)) {
          setPredictionError(
            `"${feature}" must contain a valid number.`
          );
          return;
        }

        features[feature] = numericValue;
      } else {
        features[feature] = value;
      }
    }

    setPredicting(true);

    try {
      const response = await axios.post(
        `${API_URL}/api/models/${selectedModelId}/predict`,
        {
          features,
        }
      );

      setPredictionResult(response.data);

      await Promise.all([
        fetchHistory(),
        fetchAnalytics(),
      ]);
    } catch (err) {
      const detail = err.response?.data?.detail;

      setPredictionError(
        typeof detail === "string"
          ? detail
          : Array.isArray(detail)
            ? detail.map((item) => item.msg).join(", ")
            : "Prediction failed. Check the backend."
      );
    } finally {
      setPredicting(false);
    }
  }

  // ----------------------------------------
  // DISPLAY PREDICTION LABEL
  // ----------------------------------------

  function getPredictionLabel(result) {
    if (!result) return "";

    const model = models.find(
      (item) => String(item.id) === String(result.model_id)
    );

    const modelName = model?.model_name?.toLowerCase() || "";
    const prediction = result.prediction;

    if (
      modelName.includes("iris") &&
      typeof prediction === "number"
    ) {
      const irisClasses = {
        0: "Iris setosa",
        1: "Iris versicolor",
        2: "Iris virginica",
      };

      return irisClasses[prediction] || String(prediction);
    }

    return typeof prediction === "object"
      ? JSON.stringify(prediction)
      : String(prediction);
  }

  // ----------------------------------------
  // FORMAT HISTORY DATA
  // ----------------------------------------

  function formatHistoryPrediction(record) {
    const modelName = record.model_name?.toLowerCase() || "";
    const prediction = record.prediction;

    if (
      modelName.includes("iris") &&
      typeof prediction === "number"
    ) {
      const irisClasses = {
        0: "Iris setosa",
        1: "Iris versicolor",
        2: "Iris virginica",
      };

      const label = irisClasses[prediction];

      if (label) return `${label} (${prediction})`;
    }

    return typeof prediction === "object"
      ? JSON.stringify(prediction)
      : String(prediction);
  }

  function formatDate(dateValue) {
    if (!dateValue) return "—";

    const date = new Date(dateValue);

    if (Number.isNaN(date.getTime())) {
      return String(dateValue);
    }

    return date.toLocaleString();
  }

  function formatInputFeatures(features) {
    if (!features) return "—";

    if (typeof features === "string") {
      try {
        features = JSON.parse(features);
      } catch {
        return features;
      }
    }

    if (Array.isArray(features)) {
      return features.join(", ");
    }

    return Object.entries(features)
      .map(([key, value]) => `${key}: ${value}`)
      .join(", ");
  }

  // ----------------------------------------
  // DASHBOARD
  // ----------------------------------------

  const validModelCount = validModels.length;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">ML</div>

          <div>
            <h2>ModelCloud</h2>
            <p>ML Serving Platform</p>
          </div>
        </div>

        <div className="sidebar-label">WORKSPACE</div>

        <a className="nav-item active" href="#dashboard">
          <span>▦</span> Dashboard
        </a>

        <a className="nav-item" href="#register">
          <span>＋</span> Register Model
        </a>

        <a className="nav-item" href="#train">
          <span>▤</span> Train from CSV
        </a>

        <a className="nav-item" href="#predict">
          <span>⚡</span> Run Prediction
        </a>

        <a className="nav-item" href="#models">
          <span>◫</span> Model Registry
        </a>

        <a className="nav-item" href="#history">
          <span>◷</span> Prediction History
        </a>

        <a className="nav-item" href="#analytics">
          <span>▥</span> Analytics
        </a>

        <div className="sidebar-footer">
          <div className="status-dot"></div>

          <div>
            <strong>Local Environment</strong>
            <p>API connection enabled</p>
          </div>
        </div>
      </aside>

      <main className="main-content" id="dashboard">
        <header className="topbar">
          <div>
            <p className="breadcrumb">Workspace / Overview</p>

            <h1>ML Model Dashboard</h1>

            <p className="subtitle">
              Register, validate, and run predictions with
              machine learning models.
            </p>
          </div>

          <div className="environment-badge">
            <span className="status-dot"></span>
            Development
          </div>
        </header>

        {/* DASHBOARD STATISTICS */}

        <section className="stats-grid">
          <article className="stat-card">
            <div className="stat-heading">Total Models</div>

            <div className="stat-value">
              {loadingModels ? "—" : models.length}
            </div>

            <div className="stat-description">
              Registered model artifacts
            </div>
          </article>

          <article className="stat-card">
            <div className="stat-heading">Validated Models</div>

            <div className="stat-value">
              {loadingModels ? "—" : validModelCount}
            </div>

            <div className="stat-description">
              Passed model validation
            </div>
          </article>
        </section>

        {/* PREDICTION ANALYTICS */}

        <section className="stats-grid analytics-stats">
          <article className="stat-card">
            <div className="stat-heading">Total Predictions</div>

            <div className="stat-value">
              {loadingAnalytics
                ? "..."
                : analytics?.total_predictions ?? 0}
            </div>

            <div className="stat-description">
              Predictions recorded in the database
            </div>
          </article>

          <article className="stat-card">
            <div className="stat-heading">Successful Predictions</div>

            <div className="stat-value">
              {loadingAnalytics
                ? "..."
                : analytics?.successful_predictions ?? 0}
            </div>

            <div className="stat-description">
              Completed successfully
            </div>
          </article>

          <article className="stat-card">
            <div className="stat-heading">Failed Predictions</div>

            <div className="stat-value">
              {loadingAnalytics
                ? "..."
                : analytics?.failed_predictions ?? 0}
            </div>

            <div className="stat-description">
              Recorded unsuccessful predictions
            </div>
          </article>

          <article className="stat-card">
            <div className="stat-heading">Success Rate</div>

            <div className="stat-value">
              {loadingAnalytics
                ? "..."
                : `${analytics?.success_rate ?? 0}%`}
            </div>

            <div className="stat-description">
              Successful predictions / all recorded predictions
            </div>
          </article>
        </section>

        {analyticsError && (
          <div className="alert alert-error" role="alert">
            {analyticsError}
          </div>
        )}

        <section className="analytics-grid" id="analytics">
          <article className="content-card analytics-card">
            <div className="section-heading">
              <div>
                <h2>Prediction Distribution</h2>
                <p>Number of predictions by output class.</p>
              </div>
            </div>

            {loadingAnalytics ? (
              <p className="empty-state">Loading chart...</p>
            ) : analytics?.prediction_distribution?.length ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart
                  data={analytics.prediction_distribution}
                  margin={{
                    top: 10,
                    right: 10,
                    left: 0,
                    bottom: 35,
                  }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="name"
                    angle={-15}
                    textAnchor="end"
                    interval={0}
                    height={65}
                    fontSize={11}
                  />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" name="Predictions" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="empty-state">
                No prediction data is available yet. Run a prediction
                to populate this chart.
              </p>
            )}
          </article>

          <article className="content-card analytics-card">
            <div className="section-heading">
              <div>
                <h2>Model Usage</h2>
                <p>Predictions made by each registered model.</p>
              </div>
            </div>

            {loadingAnalytics ? (
              <p className="empty-state">Loading chart...</p>
            ) : analytics?.model_usage?.length ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart
                  data={analytics.model_usage}
                  margin={{
                    top: 10,
                    right: 10,
                    left: 0,
                    bottom: 35,
                  }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="name"
                    angle={-15}
                    textAnchor="end"
                    interval={0}
                    height={65}
                    fontSize={11}
                  />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" name="Predictions" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="empty-state">
                No model usage data is available yet.
              </p>
            )}
          </article>

          <div className="analytics-actions">
            <button
              type="button"
              className="secondary-button"
              onClick={fetchAnalytics}
              disabled={loadingAnalytics}
            >
              {loadingAnalytics ? "Refreshing..." : "Refresh Analytics"}
            </button>
          </div>
        </section>

        {error && (
          <div className="alert alert-error" role="alert">
            {error}
          </div>
        )}

        {message && (
          <div className="alert alert-success" role="status">
            {message}
          </div>
        )}

        {/* REGISTER MODEL */}

        <section className="content-card" id="register">
          <div className="section-heading">
            <div>
              <h2>Register a Model</h2>
              <p>
                Upload a trained model and provide its metadata.
              </p>
            </div>

            <span className="section-tag">MODEL UPLOAD</span>
          </div>

          <form onSubmit={handleSubmit}>
            <div className="form-grid">
              <div className="form-group">
                <label htmlFor="model_name">Model Name *</label>

                <input
                  id="model_name"
                  name="model_name"
                  value={form.model_name}
                  onChange={handleChange}
                  placeholder="e.g. Iris Classifier"
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="version">Version *</label>

                <input
                  id="version"
                  name="version"
                  value={form.version}
                  onChange={handleChange}
                  placeholder="1.0.0"
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="framework">Framework *</label>

                <select
                  id="framework"
                  name="framework"
                  value={form.framework}
                  onChange={handleChange}
                  required
                >
                  <option value="scikit-learn">scikit-learn</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="output_description">
                  Output Description *
                </label>

                <input
                  id="output_description"
                  name="output_description"
                  value={form.output_description}
                  onChange={handleChange}
                  placeholder="Describe the model output"
                  required
                />
              </div>

              <div className="form-group full-width">
                <label htmlFor="description">Description</label>

                <textarea
                  id="description"
                  name="description"
                  value={form.description}
                  onChange={handleChange}
                  placeholder="Describe the purpose of your model"
                  rows="3"
                />
              </div>

              <div className="form-group full-width">
                <label htmlFor="input_schema">Input Schema (JSON) *</label>

                <textarea
                  id="input_schema"
                  name="input_schema"
                  value={form.input_schema}
                  onChange={handleChange}
                  rows="6"
                  className="code-input"
                  required
                />

                <small>
                  Feature names and order must match your trained model.
                </small>
              </div>

              <div className="form-group full-width">
                <label htmlFor="model-file">
                  Trained Model File (.joblib, .pkl, .pickle) *
                </label>

                <input
                  id="model-file"
                  type="file"
                  accept=".joblib,.pkl,.pickle"
                  onChange={(event) =>
                    setModelFile(event.target.files?.[0] || null)
                  }
                  required
                />

                <small>Upload a trained model artifact.</small>
              </div>
            </div>

            <div className="form-actions">
              <button
                className="primary-button"
                type="submit"
                disabled={loading}
              >
                {loading ? "Registering..." : "Upload & Validate Model"}
              </button>
            </div>
          </form>
        </section>

        {/* TRAIN MODEL FROM CSV */}

        <section className="content-card" id="train">
          <div className="section-heading">
            <div>
              <h2>Train a Model from CSV</h2>
              <p>
                Upload a dataset to train and register a machine
                learning model automatically.
              </p>
            </div>

            <span className="section-tag">DATASET TRAINING</span>
          </div>

          <form onSubmit={handleDatasetTraining}>
            <div className="form-grid">
              <div className="form-group full-width">
                <label htmlFor="dataset-file">CSV Dataset *</label>

                <input
                  id="dataset-file"
                  type="file"
                  accept=".csv,text/csv"
                  onChange={(event) => {
                    const file = event.target.files?.[0] || null;
                    setDatasetFile(file);
                    handleDatasetInspection(file);
                  }}
                  required
                />

                <small>
                  Upload a CSV file containing feature columns and
                  the column you want the model to predict.
                </small>
              </div>

              {inspectingDataset && (
                <div className="form-group full-width">
                  <p className="empty-state">
                    Inspecting CSV dataset...
                  </p>
                </div>
              )}

              {inspectionError && (
                <div className="form-group full-width">
                  <div className="alert alert-error" role="alert">
                    {inspectionError}
                  </div>
                </div>
              )}

              {datasetInfo && (
                <div className="form-group full-width">
                  <h3>Dataset Inspection Results</h3>

                  <p>
                    Rows: {datasetInfo.rows} | Columns:{" "}
                    {datasetInfo.column_count}
                  </p>

                  <div className="table-wrapper">
                    <table>
                      <thead>
                        <tr>
                          <th>Column</th>
                          <th>Data Type</th>
                          <th>Missing Values</th>
                          <th>Unique Values</th>
                        </tr>
                      </thead>

                      <tbody>
                        {datasetInfo.columns.map((column) => (
                          <tr key={column.name}>
                            <td>{column.name}</td>
                            <td>{column.data_type}</td>
                            <td>{column.missing_values}</td>
                            <td>{column.unique_values}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              <div className="form-group">
                <label htmlFor="dataset-model-name">
                  Model Name *
                </label>

                <input
                  id="dataset-model-name"
                  value={datasetModelName}
                  onChange={(event) =>
                    setDatasetModelName(event.target.value)
                  }
                  placeholder="e.g. Fruit Classifier"
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="target-column">Target Column *</label>

                <select
                  id="target-column"
                  value={targetColumn}
                  onChange={(event) => {
                    const colName = event.target.value;
                    setTargetColumn(colName);

                    const selectedColumn = datasetInfo?.columns?.find(
                      (column) => column.name === colName
                    );

                    if (selectedColumn) {
                      if (
                        colName === datasetInfo.suggested_target &&
                        datasetInfo.suggested_task
                      ) {
                        setTrainingTask(datasetInfo.suggested_task);
                      } else if (selectedColumn.data_type === "numeric") {
                        if (
                          selectedColumn.unique_values <= 10 &&
                          selectedColumn.unique_values < (datasetInfo.rows || 100) * 0.2
                        ) {
                          setTrainingTask("classification");
                        } else {
                          setTrainingTask("regression");
                        }
                      } else {
                        setTrainingTask("classification");
                      }
                    }
                  }}
                  required
                  disabled={!datasetInfo}
                >
                  <option value="">Select target column</option>

                  {datasetInfo?.columns?.map((column) => (
                    <option key={column.name} value={column.name}>
                      {column.name}
                      {column.name === datasetInfo.suggested_target
                        ? " (Suggested)"
                        : ""}
                    </option>
                  ))}
                </select>

                <small>
                  {datasetInfo
                    ? `Suggested target: ${datasetInfo.suggested_target || "None"}. You can change it.`
                    : "Upload a CSV file to detect available columns."}
                </small>
              </div>

              <div className="form-group full-width">
                <label htmlFor="training-task">Training Task *</label>

                <select
                  id="training-task"
                  value={trainingTask}
                  onChange={(event) =>
                    setTrainingTask(event.target.value)
                  }
                  required
                  disabled={!datasetInfo}
                >
                  <option value="classification">Classification</option>
                  <option value="regression">Regression</option>
                </select>

                <small>
                  {datasetInfo?.suggested_task
                    ? `Backend suggestion: ${datasetInfo.suggested_task}. You can override this selection.`
                    : "Classification predicts categories; regression predicts numeric values."}
                </small>
              </div>
            </div>

            {trainingError && (
              <div className="alert alert-error" role="alert">
                {trainingError}
              </div>
            )}

            {trainingMessage && (
              <div className="alert alert-success" role="status">
                {trainingMessage}
              </div>
            )}

            {trainingMetrics && (
              <div className="metrics-card">
                <div className="metrics-card-header">
                  <h3>Model Evaluation Results</h3>
                  <span className="status-badge valid">EVALUATION COMPLETE</span>
                </div>

                <div className="metrics-summary-grid">
                  {trainingMetrics.accuracy !== undefined ? (
                    <>
                      <div className="metric-box">
                        <span className="metric-box-label">Accuracy</span>
                        <strong className="metric-box-val">
                          {(trainingMetrics.accuracy * 100).toFixed(2)}%
                        </strong>
                      </div>
                      <div className="metric-box">
                        <span className="metric-box-label">Precision (Weighted)</span>
                        <strong className="metric-box-val">
                          {trainingMetrics.precision !== undefined
                            ? (trainingMetrics.precision * 100).toFixed(2) + "%"
                            : "—"}
                        </strong>
                      </div>
                      <div className="metric-box">
                        <span className="metric-box-label">Recall (Weighted)</span>
                        <strong className="metric-box-val">
                          {trainingMetrics.recall !== undefined
                            ? (trainingMetrics.recall * 100).toFixed(2) + "%"
                            : "—"}
                        </strong>
                      </div>
                      <div className="metric-box">
                        <span className="metric-box-label">F1 Score</span>
                        <strong className="metric-box-val">
                          {(trainingMetrics.f1_weighted * 100).toFixed(2)}%
                        </strong>
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="metric-box">
                        <span className="metric-box-label">MAE</span>
                        <strong className="metric-box-val">{trainingMetrics.mae}</strong>
                      </div>
                      <div className="metric-box">
                        <span className="metric-box-label">MSE</span>
                        <strong className="metric-box-val">
                          {trainingMetrics.mse ?? "—"}
                        </strong>
                      </div>
                      <div className="metric-box">
                        <span className="metric-box-label">RMSE</span>
                        <strong className="metric-box-val">{trainingMetrics.rmse}</strong>
                      </div>
                      <div className="metric-box">
                        <span className="metric-box-label">R² Score</span>
                        <strong className="metric-box-val">
                          {trainingMetrics.r2 !== null && trainingMetrics.r2 !== undefined
                            ? trainingMetrics.r2
                            : "N/A"}
                        </strong>
                      </div>
                    </>
                  )}
                </div>

                <div className="dataset-split-info">
                  <span><strong>Training Samples:</strong> {trainingMetrics.training_rows ?? "—"}</span>
                  <span><strong>Testing Samples:</strong> {trainingMetrics.test_rows}</span>
                </div>

                {trainingMetrics.confusion_matrix && (
                  <div className="confusion-matrix-container">
                    <h4>Confusion Matrix</h4>
                    <table className="confusion-matrix-table">
                      <thead>
                        <tr>
                          <th>Actual \ Predicted</th>
                          {(trainingMetrics.classes || trainingMetrics.confusion_matrix.map((_, i) => `Class ${i}`)).map((cls) => (
                            <th key={cls}>{cls}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {trainingMetrics.confusion_matrix.map((row, rIdx) => (
                          <tr key={rIdx}>
                            <td><strong>{(trainingMetrics.classes && trainingMetrics.classes[rIdx]) || `Class ${rIdx}`}</strong></td>
                            {row.map((cell, cIdx) => (
                              <td key={cIdx} className={rIdx === cIdx ? "cm-diagonal" : "cm-off"}>
                                {cell}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}

            <div className="form-actions">
              <button
                className="primary-button"
                type="submit"
                disabled={training || inspectingDataset || !datasetInfo}
              >
                {training ? "Training Model..." : "Upload CSV & Train Model"}
              </button>
            </div>
          </form>
        </section>

        {/* RUN PREDICTION */}

        <section className="content-card" id="predict">
          <div className="section-heading">
            <div>
              <h2>Run Prediction</h2>

              <p>
                Enter feature values and get a prediction from the
                selected model.
              </p>
            </div>

            <span className="section-tag">MODEL INFERENCE</span>
          </div>

          {validModels.length === 0 ? (
            <div className="empty-state">
              No validated models are available. Register a compatible
              model first.
            </div>
          ) : (
            <form onSubmit={handlePredict}>
              <div className="form-grid">
                <div className="form-group full-width">
                  <label htmlFor="prediction-model">
                    Select Validated Model *
                  </label>

                  <select
                    id="prediction-model"
                    value={selectedModelId}
                    onChange={(event) =>
                      setSelectedModelId(event.target.value)
                    }
                    required
                  >
                    {validModels.map((model) => (
                      <option
                        key={model.id}
                        value={String(model.id)}
                      >
                        {model.model_name} — v{model.version} (ID:{" "}
                        {model.id})
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {loadingSchema ? (
                <p className="empty-state">
                  Loading model input features...
                </p>
              ) : predictionSchema.length > 0 ? (
                <>
                  <div className="prediction-fields">
                    {predictionSchema.map((featureDefinition) => {
                      const feature = featureDefinition.name;
                      const isNumeric =
                        featureDefinition.type === "numeric";

                      return (
                        <div
                          className="form-group"
                          key={feature}
                        >
                          <label htmlFor={`prediction-${feature}`}>
                            {feature.replaceAll("_", " ")}
                            {isNumeric ? " (Number)" : " (Text)"}
                          </label>

                          <input
                            id={`prediction-${feature}`}
                            type={isNumeric ? "number" : "text"}
                            step={isNumeric ? "any" : undefined}
                            value={predictionValues[feature] ?? ""}
                            onChange={(event) =>
                              handlePredictionValueChange(
                                feature,
                                event.target.value
                              )
                            }
                            placeholder={`Enter ${feature}`}
                            required
                          />
                        </div>
                      );
                    })}
                  </div>

                  <div className="form-actions">
                    <button
                      className="primary-button"
                      type="submit"
                      disabled={predicting || loadingSchema}
                    >
                      {predicting ? "Predicting..." : "Run Prediction"}
                    </button>

                    <button
                      className="secondary-button"
                      type="button"
                      onClick={() => {
                        setPredictionResult(null);
                        setPredictionError("");
                      }}
                      disabled={predicting}
                    >
                      Clear Result
                    </button>
                  </div>
                </>
              ) : (
                <p className="empty-state">
                  No input features could be loaded.
                </p>
              )}
            </form>
          )}

          {predictionError && (
            <div className="alert alert-error" role="alert">
              {String(predictionError)}
            </div>
          )}

          {predictionResult && (
            <div className="prediction-result" role="status">
              <div className="prediction-result-heading">
                <div>
                  <span className="prediction-eyebrow">
                    PREDICTION COMPLETED
                  </span>

                  <h3>
                    {predictionResult.model_name || "Model prediction"}
                  </h3>
                </div>

                <span className="status-badge valid">SUCCESS</span>
              </div>

              <div className="prediction-output">
                <span>Predicted Output</span>

                <strong>
                  {getPredictionLabel(predictionResult)}
                </strong>
              </div>

              {predictionResult.output_description && (
                <p className="prediction-description">
                  {predictionResult.output_description}
                </p>
              )}

              <div className="prediction-input-summary">
                <h4>Submitted Input Features</h4>

                <div className="prediction-summary-grid">
                  {Object.entries(
                    predictionResult.input_features || {}
                  ).map(([feature, value]) => (
                    <div
                      className="prediction-summary-item"
                      key={feature}
                    >
                      <span>{feature.replaceAll("_", " ")}</span>
                      <strong>{String(value)}</strong>
                    </div>
                  ))}
                </div>
              </div>

              <details className="prediction-raw">
                <summary>View raw API response</summary>

                <pre>
                  {JSON.stringify(predictionResult, null, 2)}
                </pre>
              </details>
            </div>
          )}
        </section>

        {/* MODEL REGISTRY */}

        <section className="content-card" id="models">
          <div className="section-heading">
            <div>
              <h2>Model Registry</h2>

              <p>
                Models registered with your backend API.
              </p>
            </div>

            <button
              type="button"
              className="secondary-button"
              onClick={fetchModels}
              disabled={loadingModels}
            >
              {loadingModels ? "Refreshing..." : "Refresh"}
            </button>
          </div>

          {loadingModels ? (
            <p className="empty-state">Loading registered models...</p>
          ) : models.length === 0 ? (
            <p className="empty-state">
              No models found. Upload your first model above.
            </p>
          ) : (
            <div className="table-wrapper">
              <table>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Framework</th>
                    <th>Version</th>
                    <th>Validation</th>
                    <th>Metrics</th>
                  </tr>
                </thead>

                <tbody>
                  {models.map((model) => (
                    <tr key={model.id}>
                      <td>
                        <strong>{model.model_name}</strong>

                        <div className="table-description">
                          Model ID: {model.id}
                        </div>
                      </td>

                      <td>{model.framework}</td>
                      <td>{model.version}</td>

                      
   <td>
                        <span
                          className={`status-badge ${
                            model.validation_status === "VALID"
                              ? "valid"
                              : "invalid"
                          }`}
                        >
                          {model.validation_status || "UNKNOWN"}
                        </span>
                      </td>

                      <td>
                        {model.metrics ? (
                          model.metrics.accuracy !== undefined ? (
                            <div>
                              <strong>Accuracy: {(model.metrics.accuracy * 100).toFixed(1)}%</strong>
                              <div className="table-description">
                                F1: {(model.metrics.f1_weighted * 100).toFixed(1)}%
                                {model.metrics.precision !== undefined ? ` | P: ${(model.metrics.precision * 100).toFixed(1)}%` : ""}
                                {model.metrics.recall !== undefined ? ` | R: ${(model.metrics.recall * 100).toFixed(1)}%` : ""}
                              </div>
                            </div>
                          ) : (
                            <div>
                              <strong>R²: {model.metrics.r2 ?? "N/A"} | RMSE: {model.metrics.rmse}</strong>
                              <div className="table-description">
                                MAE: {model.metrics.mae}{model.metrics.mse !== undefined ? ` | MSE: ${model.metrics.mse}` : ""}
                              </div>
                            </div>
                          )
                        ) : (
                          "N/A"
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        {/* PREDICTION HISTORY */}

        <section className="content-card" id="history">
          <div className="section-heading">
            <div>
              <h2>Prediction History</h2>

              <p>
                Review previously saved predictions from your registered
                models.
              </p>
            </div>

            <button
              type="button"
              className="secondary-button"
              onClick={fetchHistory}
              disabled={loadingHistory}
            >
              {loadingHistory ? "Refreshing..." : "Refresh History"}
            </button>
          </div>

          {historyError && (
            <div className="alert alert-error" role="alert">
              {historyError}
            </div>
          )}

          {loadingHistory && history.length === 0 ? (
            <p className="empty-state">
              Loading prediction history...
            </p>
          ) : history.length === 0 ? (
            <div className="empty-state">
              <h3>No prediction history yet</h3>

              <p>
                Run a prediction above. Successful predictions will
                appear here automatically.
              </p>
            </div>
          ) : (
            <>
              <div className="history-summary">
                <div>
                  <strong>{history.length}</strong>
                  <span>Records loaded</span>
                </div>

                <div>
                  <strong>
                    {
                      history.filter(
                        (record) => record.status === "SUCCESS"
                      ).length
                    }
                  </strong>

                  <span>Successful predictions</span>
                </div>
              </div>

              <div className="table-wrapper">
                <table className="history-table">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Model</th>
                      <th>Version</th>
                      <th>Input Features</th>
                      <th>Prediction</th>
                      <th>Status</th>
                      <th>Timestamp</th>
                    </tr>
                  </thead>

                  <tbody>
                    {history.map((record) => (
                      <tr key={record.id}>
                        <td>#{record.id}</td>

                        <td>
                          <strong>{record.model_name}</strong>

                          <div className="table-description">
                            Model ID: {record.model_id}
                          </div>
                        </td>

                        <td>{record.model_version}</td>

                        <td className="history-input">
                          {formatInputFeatures(record.input_features)}
                        </td>

                        <td>
                          <strong>
                            {formatHistoryPrediction(record)}
                          </strong>
                        </td>

                        <td>
                          <span
                            className={`status-badge ${
                              record.status === "SUCCESS"
                                ? "valid"
                                : "invalid"
                            }`}
                          >
                            {record.status}
                          </span>
                        </td>

                        <td className="history-date">
                          {formatDate(record.created_at)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </section>

        <footer className="page-footer">
          Cloud-Based ML Model Serving Platform · Development Build
        </footer>
      </main>
    </div>
  );
}

export default App;