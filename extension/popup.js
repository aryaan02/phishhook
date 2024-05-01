chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
  chrome.scripting.executeScript(
    {
      target: { tabId: tabs[0].id },
      function: listLinks,
    },
    (injectionResults) => {
      const linksContainer = document.getElementById("linkList");
      for (const frameResult of injectionResults) {
        const links = frameResult.result;
        links.forEach((url) => {
          const li = document.createElement("li");
          const textNode = document.createTextNode(url + " - ");
          li.appendChild(textNode);

          // Simulate fetching data for each link
          simulateFetch(url).then((data) => {
            const responseText = `Probability: ${data.probability}, Is Phishing: ${data.is_phishing}`;
            const responseNode = document.createElement("span");
            responseNode.textContent = responseText;
            li.appendChild(responseNode);
          });

          linksContainer.appendChild(li);
        });
      }
    }
  );
});

function listLinks() {
  const links = document.getElementsByTagName("a");
  return Array.from(links).map((link) => link.href);
}

function simulateFetch(link) {
  return new Promise((resolve) => {
    setTimeout(() => {
      const mockData = {
        probability: Math.random().toFixed(2), // Random probability for demonstration
        is_phishing: Math.random() > 0.5 ? "Yes" : "No", // Randomly decide if phishing or not
      };
      resolve(mockData);
    }, 500); // Simulate a shorter delay
  });
}
