setInterval(() => {
    const cuotas = document.querySelectorAll(".payout");

    if (cuotas.length === 0) return;

    const ultima = cuotas[0].innerText.replace("x", "").trim();

    fetch("https://aviator-ia-iw6r.onrender.com/api/cuota", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            cuota: parseFloat(ultima)
        })
    })
    .then(r => r.json())
    .then(console.log)
    .catch(console.error);

}, 1000);
