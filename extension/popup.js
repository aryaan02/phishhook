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
            const responseText = `Probability: ${data.probability.toFixed(2) * 100}%, Is Phishing: ${data.is_phishing}`;
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
  return fetch(`http://localhost:5000/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ url: link })
  })
    .then(response => {
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      return response.json();
    })
    .catch(error => {
      console.error('There has been a problem with your fetch operation:', error);
    });
}

