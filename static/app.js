let get_input_label = function() {
    let my_pic = canvas.toDataURL();
    return {'JSON_pic': my_pic}
};


let send_coefficient_json = function(coefficients) {
    $.ajax({
        url: '/classify',
        contentType: "application/json; charset=utf-8",
        type: 'POST',
        success: function (data) {
            display_solutions(data);
        },
        data: JSON.stringify(coefficients)
    });
};

let display_solutions = function(solutions) {
    $("span#solution1").html(solutions.model_output)
    $("img#photo2").attr("src",solutions.pic_x)
};


$(document).ready(function() {

    $("button#classify").click(function() {
        let coefficients = get_input_label();
        send_coefficient_json(coefficients);
    })

})
