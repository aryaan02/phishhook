const links = document.getElementsByTagName("a");
Array.from(links).forEach((link) => {
  simulateFetch(link.href).then((data) => {
    if (data.is_phishing === "Yes") {
      // Apply basic styling to the link
      link.style.color = "red";
      link.style.fontWeight = "bold";

      // Create and style the tooltip
      const tooltip = document.createElement("span");
      tooltip.textContent = "Warning: Phishing Link!";
      tooltip.style.visibility = "hidden";
      tooltip.style.width = "140px";
      tooltip.style.backgroundColor = "black";
      tooltip.style.color = "#fff";
      tooltip.style.textAlign = "center";
      tooltip.style.borderRadius = "6px";
      tooltip.style.padding = "5px 0";
      tooltip.style.position = "absolute";
      tooltip.style.zIndex = "1";
      tooltip.style.bottom = "150%";
      tooltip.style.left = "50%";
      tooltip.style.marginLeft = "-75px";
      tooltip.style.boxShadow = "0px 0px 6px 0px rgba(0,0,0,0.75)";

      // Tooltip arrow
      const arrow = document.createElement("span");
      arrow.style.visibility = "hidden";
      arrow.style.position = "absolute";
      arrow.style.top = "100%";
      arrow.style.left = "50%";
      arrow.style.marginLeft = "-5px";
      arrow.style.borderWidth = "5px";
      arrow.style.borderStyle = "solid";
      arrow.style.borderColor = "black transparent transparent transparent";

      link.appendChild(tooltip);
      link.appendChild(arrow);

      // Show tooltip on hover
      link.onmouseover = function () {
        tooltip.style.visibility = "visible";
        arrow.style.visibility = "visible";
      };
      link.onmouseout = function () {
        tooltip.style.visibility = "hidden";
        arrow.style.visibility = "hidden";
      };
    }
  });
});

function simulateFetch(url) {
  return fetch('http://localhost:5000/predict', {  // Adjust the URL/port as necessary
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({url: url})
  })
  .then(response => response.json())
  .then(data => {
    return {
      probability: data.probability.toFixed(2),
      is_phishing: data.is_phishing === "Yes" ? "Yes" : "No"
    };
  })
  .catch(error => {
    console.error('Error:', error);
    return {
      probability: "0.00",
      is_phishing: "No"
    };
  });
}

