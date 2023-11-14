function getRecommendations() {
    const userId = document.getElementById('userId').value;

    fetch('/get_recommendations', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: `userId=${userId}`
    })
    .then(response => response.json())
    .then(data => displayRecommendations(data));
}

function displayRecommendations(recommendations) {
    const recommendationsDiv = document.getElementById('recommendations');
    recommendationsDiv.innerHTML = '<h3>Recomendaciones:</h3>';
    
    recommendations.forEach(movie => {
        const movieDiv = document.createElement('div');
        movieDiv.textContent = movie;
        recommendationsDiv.appendChild(movieDiv);
    });
}
