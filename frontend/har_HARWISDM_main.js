"use strict";
console.log('this is har_main.js');
// var str = "";
// for(var i=0;i<1152;i++){
//     str += i + ","
// }
// str += 1152;
// console.log(str);
var id = 210;
var dimensionIndex = 3;
var ecgData = [];
var scores = [];
var suportColor = d3.rgb(255, 255, 255);
var blameColor = d3.rgb(0, 0, 0);
var compute = d3.interpolate(suportColor, blameColor);
var colorLinear;
var maxAndMin = -9999;
var absMax = 0;
var oneDimensionArray = [];
// var activityDict = {
//     "1": "Walking",
//     "2": "LyingDown",
//     "3": "Standing",
//     "4": "Sitting",
//     "5": "Jogging",
//     "6": "Stairs"
// };
var activityDict = {
    "1": "Jogging",
    "2": "LyingDown",
    "3": "Sitting",
    "4": "Stairs",
    "5": "Standing",
    "6": "Walking"
};
var dimensionIndexDict = {
    "1": "accelerometter_x",
    "2": "accelerometter_y",
    "3": "accelerometter_z"
};
var colors = [
    'black',
    'purple',
    'steelblue',
    'green',
    'pink',
    'brown',
    'red'
];
// d3.csv('./data/HAR/raw_all_HARWISDM_notNormalized_yval.csv', function (res) {
//     console.log(res);
//     var count = [0,0,0,0,0,0];
//     var sum = 0;
//     for (var i=0; i<res.length;i++) {
//         for(var k in res[i]) {
//             if (res[i][k] == 1) {
//                 count[k]++
//             }
//         }
//     }
//     sum = count[0] + count[1] + count[2] + count[3] + count[4] + count[5];
//     console.log(count);
//     console.log(count[0]/sum, count[1]/sum, count[2]/sum, count[3]/sum, count[4]/sum, count[5]/sum);
// })
d3.csv('./data/HAR/raw_all_HARWISDM_notNormalized_Xval.csv', function (res) {
    console.log(res);
    var scale1 = 10,
        scale2 = 10;
    pushRecord2Array(res, 1, colors[0]);
    d3.csv('./data/HAR/raw_all_HARWISDM_notNormalized_final_scores_0.csv', function (fs0) {
        pushRecord2Array(fs0, scale1, colors[1]);
        pushRecord2OneDArray(fs0, scale2, 0, colors[1]);
        d3.csv('./data/HAR/raw_all_HARWISDM_notNormalized_final_scores_1.csv', function (fs1) {
            pushRecord2Array(fs1, scale1, colors[2]);
            pushRecord2OneDArray(fs1, scale2, 1, colors[2]);
            console.log(fs1);
            d3.csv('./data/HAR/raw_all_HARWISDM_notNormalized_final_scores_2.csv', function (fs2) {
                pushRecord2Array(fs2, scale1, colors[3]);
                pushRecord2OneDArray(fs2, scale2, 2, colors[3]);
                console.log(fs2);
                d3.csv('./data/HAR/raw_all_HARWISDM_notNormalized_final_scores_3.csv', function (fs3) {
                    pushRecord2Array(fs3, scale1, colors[4]);
                    pushRecord2OneDArray(fs3, scale2, 3, colors[4]);
                    console.log(fs3);
                    d3.csv('./data/HAR/raw_all_HARWISDM_notNormalized_final_scores_4.csv', function (fs4) {
                        pushRecord2Array(fs4, scale1, colors[5]);
                        pushRecord2OneDArray(fs4, scale2, 4, colors[5]);
                        console.log(fs4);
                        d3.csv('./data/HAR/raw_all_HARWISDM_notNormalized_final_scores_5.csv', function (fs5) {
                            pushRecord2Array(fs5, scale1, colors[6]);
                            pushRecord2OneDArray(fs5, scale2, 5, colors[6]);
                            console.log(scores);
                            maxAndMin = getMaxAndMin(oneDimensionArray);
                            absMax = Math.abs(maxAndMin.max) > Math.abs(maxAndMin.min) ? Math.abs(maxAndMin.max) : Math.abs(maxAndMin.min);
                            console.log(absMax);
                            colorLinear = d3.scale.linear().domain([-absMax, absMax]).range([0, 1]);
                            console.log(maxAndMin);
                            console.log(fs5);
                            drawGraph(ecgData);
                        })
                    })
                })
            })
        })
    });
});
function getMaxAndMin(array) {
    var max = -999, min = 999;
    array.forEach(function (a) {
        console.log(a);
        a.forEach(function (n) {
            var v = n.value;
            max = v > max ? v : max;
            min = v < min ? v : min;
        })
    });
    return {
        max: max,
        min: min
    }
}
function pushRecord2Array(record, scale, color) {
    var ecgRecord = [];
    // console.log(fs0);
    var count = 1;
    var numberIndex = 1;
    for(var k in record[id]) {
        var point = {
            x: numberIndex,
            y: Number(record[id][k] * scale),
            color: color
        };
        if ((count + 3 - dimensionIndex) % 3 === 0) {
            ecgRecord.push(point);
            numberIndex++;
        }
        count++;
    }
    ecgData.push(ecgRecord);
}

function pushRecord2OneDArray(record, scale, index, color) {
    var ecgRecord = [];
    // console.log(fs0);
    var count = 1;
    var numberIndex = 1;
    for(var k in record[id]) {
        var point = {
            x: numberIndex,
            y: 15 + index,
            r: Number(record[id][k] * scale),
            index: index,
            value: Number(record[id][k]),
            color: color
        };
        if ((count + 3 - dimensionIndex) % 3 === 0) {
            ecgRecord.push(point);
            numberIndex++;
        }
        count++;
    }
    oneDimensionArray.push(ecgRecord);
    ecgData.push(ecgRecord);
}





function drawGraph(data) {
    //************************************************************
// Create Margins and Axis and hook our zoom function
//************************************************************
    var margin = {top: 30, right: 30, bottom: 30, left: 50},
        width = 960 * 1.5 - margin.left - margin.right,
        height = 500 * 1.5 + 20 - margin.top - margin.bottom;

    var x = d3.scale.linear()
        .domain([0, 90])
        .range([0, width]);

    var y = d3.scale.linear()
        .domain([-20, 20])
        .range([height, 0]);

    var xAxis = d3.svg.axis()
        .scale(x)
        .tickSize(-height)
        .tickPadding(10)
        .tickSubdivide(true)
        .orient("bottom");

    var yAxis = d3.svg.axis()
        .scale(y)
        .tickPadding(10)
        .tickSize(-width)
        .tickSubdivide(true)
        .orient("left");

    var zoom = d3.behavior.zoom()
        .x(x)
        .y(y)
        .scaleExtent([1, 10])
        .on("zoom", zoomed);





//************************************************************
// Generate our SVG object
//************************************************************
    var svg = d3.select("body").append("svg")
        .call(zoom)
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    svg.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + height + ")")
        .call(xAxis);

    svg.append("g")
        .attr("class", "y axis")
        .call(yAxis);

    svg.append("g")
        .attr("class", "y axis")
        .append("text")
        .attr("class", "axis-label")
        .attr("transform", "rotate(-90)")
        .attr("y", (-margin.left) + 10)
        .attr("x", -height/2)
        .text('Support Scores');

    svg.append("clipPath")
        .attr("id", "clip")
        .append("rect")
        .attr("width", width)
        .attr("height", height);





//************************************************************
// Create D3 line object and draw data on our SVG object
//************************************************************
    var line = d3.svg.line()
        .interpolate("linear")
        .x(function(d) { return x(d.x); })
        .y(function(d) { return y(d.y); });

    svg.selectAll('.line')
        .data(data)
        .enter()
        .append("path")
        .attr("class", "line")
        .attr("id", function (d, i) {
            return 'tag' + i;
        })
        .attr("clip-path", "url(#clip)")
        .attr('stroke', function(d,i){
            console.log(d);
            return d[0].color;
        })
        .attr("d", line);

    var legend = [{
        key: 'HAR data',
        active: true
    }, {
        key: activityDict["1"],
        active: true
    }, {
        key: activityDict["2"],
        active: true
    }, {
        key: activityDict["3"],
        active: true
    }, {
        key: activityDict["4"],
        active: true
    }, {
        key: activityDict["5"],
        active: true
    }, {
        key: activityDict["6"],
        active: true
    }];
    var legendSpace = width;
    // Add the Legend
    legend.forEach(function (d, i) {
        svg.append("text")
            .attr("x", function () {
                return (legendSpace/2) + (i - legend.length/2) * 150
            })  // space legend
            .attr("y", height + (margin.bottom/2)+ 15)
            .attr("class", "legend")    // style the legend
            .style("fill", function() { // Add the colours dynamically
                console.log(d);
                return colors[i%colors.length]; })
            .on("click", function(){
                console.log('clicked');
                // Determine if current line is visible
                var active = d.active ? false : true,
                    newOpacity = active ? 0 : 1;
                // Hide or show the elements based on the ID
                d3.select("#tag"+d.key.replace(/\s+/g, ''))
                    .transition().duration(100)
                    .style("opacity", newOpacity);
                // Update whether or not the elements are active
                d.active = active;
            })
            .text(d.key);
    })





//************************************************************
// Draw points on SVG object based on the data given
//************************************************************
    var points = svg.selectAll('.dots')
        .data(data)
        .enter()
        .append("g")
        .attr("class", "dots")
        .attr("clip-path", "url(#clip)");

    points.selectAll('.dot')
        .data(function(d, index){
            var a = [];
            d.forEach(function(point,i){
                a.push({'index': index, 'point': point});
            });
            return a;
        })
        .enter()
        .append('circle')
        .attr('class','dot')
        .attr('stroke','black')
        .attr('stroke-width', '1px')
        .attr("r", function (d, i) {
            if(!!d.point.r) {
                return 10;
            } else {
                return 2.5;
            }
        })
        .attr('fill', function(d,i){
            // console.log(d);
            if (!!d.point.r) {
                console.log(d);
                return compute(colorLinear(d.point.value));
            } else {
                return d.point.color;
            }
            // console.log(i);
            // console.log(scores);
            // console.log(scores[0][i].y)
            // return compute(colorLinear(scores[0][i].y));
        })
        .attr("transform", function(d) {
            return "translate(" + x(d.point.x) + "," + y(d.point.y) + ")"; }
        );





//************************************************************
// Zoom specific updates
//************************************************************
    function zoomed() {
        svg.select(".x.axis").call(xAxis);
        svg.select(".y.axis").call(yAxis);
        svg.selectAll('path.line').attr('d', line);

        points.selectAll('circle').attr("transform", function(d) {
            return "translate(" + x(d.point.x) + "," + y(d.point.y) + ")"; }
        );
    }
}