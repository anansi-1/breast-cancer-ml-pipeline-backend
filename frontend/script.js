document.getElementById("predictBtn").addEventListener("click", async () => {
  const input = document.getElementById("features").value;
  const features = input.split(",").map((x) => parseFloat(x.trim()));
  const resultsDiv = document.getElementById("results");
  const messageDiv = document.getElementById("message");

  resultsDiv.innerHTML = "";
  messageDiv.textContent = "";

  if (features.length !== 30) {
    messageDiv.textContent = `Invalid number of features: expected 30, got ${features.length}`;
    return;
  }

  try {
    const response = await fetch(
      "https://breast-cancer-backend-fvx9.onrender.com/predict",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features }),
      }
    );

    const data = await response.json();

    if (!response.ok) {
      messageDiv.textContent = "Error: " + data.detail;
      return;
    }

    let output = "";
    for (const [model, prediction] of Object.entries(data)) {
      output += `${model.replace("_", " ").toUpperCase()}: ${
        prediction === 0 ? "Benign" : "Malignant"
      }\n`;
    }
    resultsDiv.textContent = output;
  } catch (err) {
    messageDiv.textContent = "Error: " + err;
  }
});
