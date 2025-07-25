document.addEventListener('DOMContentLoaded', function () {
    // Sample data for ROC Curve
    const rocData = {
        labels: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        datasets: [{
            label: 'ROC Curve',
            data: [0, 0.1, 0.4, 0.5, 0.7, 0.75, 0.85, 0.9, 0.95, 0.98, 1],
            fill: false,
            borderColor: 'rgb(75, 192, 192)',
            tension: 0.1
        }]
    };

    // Sample data for Precision-Recall Curve
    const prData = {
        labels: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        datasets: [{
            label: 'Precision-Recall Curve',
            data: [1, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45],
            fill: false,
            borderColor: 'rgb(255, 99, 132)',
            tension: 0.1
        }]
    };

    // Config for ROC Curve
    const rocConfig = {
        type: 'line',
        data: rocData,
        options: {
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'False Positive Rate'
                    },
                    min: 0,
                    max: 1
                },
                y: {
                    title: {
                        display: true,
                        text: 'True Positive Rate'
                    },
                    min: 0,
                    max: 1
                }
            }
        }
    };

    // Config for Precision-Recall Curve
    const prConfig = {
        type: 'line',
        data: prData,
        options: {
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Recall'
                    },
                    min: 0,
                    max: 1
                },
                y: {
                    title: {
                        display: true,
                        text: 'Precision'
                    },
                    min: 0,
                    max: 1
                }
            }
        }
    };

    // Render ROC Curve
    const rocCtx = document.getElementById('rocCurve').getContext('2d');
    new Chart(rocCtx, rocConfig);

    // Render Precision-Recall Curve
    const prCtx = document.getElementById('prCurve').getContext('2d');
    new Chart(prCtx, prConfig);
});
