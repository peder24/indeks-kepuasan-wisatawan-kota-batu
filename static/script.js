$(document).ready(function() {
    // Handle prediction form submission
    $('#predictionForm').on('submit', function(e) {
        e.preventDefault();
        
        var reviewText = $('#reviewText').val();
        var visitTime = $('#visitTime').val();
        
        // Show loading
        $('#predictionResult').html('<div class="alert alert-info">Memproses prediksi...</div>');
        
        // Send prediction request
        $.ajax({
            url: '/predict',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                text: reviewText,
                visit_time: visitTime
            }),
            success: function(response) {
                var resultHtml = '<div class="alert alert-success">';
                resultHtml += '<h5>Hasil Prediksi:</h5>';
                resultHtml += '<p><strong>Tingkat Kepuasan:</strong> ' + response.satisfaction + '</p>';
                resultHtml += '<p><strong>Sentimen:</strong> ' + response.sentiment + '</p>';
                resultHtml += '<p><strong>Probabilitas:</strong></p>';
                resultHtml += '<ul>';
                
                for (var key in response.probabilities) {
                    var percentage = (response.probabilities[key] * 100).toFixed(2);
                    resultHtml += '<li>' + key + ': ' + percentage + '%</li>';
                }
                
                resultHtml += '</ul>';
                resultHtml += '</div>';
                
                $('#predictionResult').html(resultHtml);
            },
            error: function() {
                $('#predictionResult').html('<div class="alert alert-danger">Terjadi kesalahan saat memproses prediksi.</div>');
            }
        });
    });
});