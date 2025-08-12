// Track coding practice sessions
function trackCodingSession(topic, difficulty, timeSpent, completed) {
    fetch('/track-coding-attempt', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            topic: topic,
            difficulty: difficulty,
            time_spent: timeSpent,
            completed: completed
        })
    }).then(response => response.json())
      .then(data => console.log('Session tracked:', data));
}

// Add to your practice.html template
let sessionStartTime = Date.now();

// Call this when user completes a problem
function onProblemComplete(topic, difficulty, success) {
    const timeSpent = Math.floor((Date.now() - sessionStartTime) / 1000);
    trackCodingSession(topic, difficulty, timeSpent, success);
    sessionStartTime = Date.now(); // Reset for next problem
}
