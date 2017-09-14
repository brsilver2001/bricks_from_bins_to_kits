let get_input_coefficients = function() {
    let JSON_input_vals = $("input#testing").val()
    return {'JSON_INPUT_KEY': parseInt(JSON_input_vals)}
};

let send_coefficient_json = function(coefficients) {
    $.ajax({
        url: '/solve',
        contentType: "application/json; charset=utf-8",
        type: 'POST',
        success: function (data) {
            display_solutions(data);
        },
        data: JSON.stringify(coefficients)
    });
};

let display_solutions = function(solutions) {
    $("span#solution").html(solutions.root_1)
};


$(document).ready(function() {

    $("button#solve").click(function() {
        let coefficients = get_input_coefficients();
        send_coefficient_json(coefficients);
    })

})
