let ultima = "";

setInterval(() => {
    const cuotas = document.querySelectorAll(".payout");

    if (cuotas.length === 0) return;

    const nueva = cuotas[0].innerText.trim();

    if (nueva !== ultima) {
        ultima = nueva;

        console.log("Nueva cuota:", nueva);

        fetch("https://aviator-ia-iw6r.onrender.com/api/cuota", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                cuota: parseFloat(nueva.replace("x", ""))
            })
        })
        .then(r => r.json())
        .then(console.log)
        .catch(console.error);
    }

}, 300);
