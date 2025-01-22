/*d3.select("input").on("keyup", function(e){
	if(e.which === 13) {
		e.preventDefault();
		d3.select("#query-button").click();	
	}
});*/

d3.select("#query-button").on("click", 
	function(){ 
		//console.log("click"); 
		var srcSelect = d3.select("#src-select").node();
		var src = srcSelect.options[srcSelect.selectedIndex].value;
		var srcName = srcSelect.options[srcSelect.selectedIndex].text;
		var tgtSelect = d3.select("#tgt-select").node();
		var tgt = tgtSelect.options[tgtSelect.selectedIndex].value;
		var tgtName = tgtSelect.options[tgtSelect.selectedIndex].text;
		var word = d3.select("#query-word-input").node().value;
		console.log(src, tgt, word);

		/*** clear elements ***/
		reset_result_panels();
		
		/*d3.select("#src-result-card").classed("invisible", true);
		d3.select("#tgt-result-card").classed("invisible", true);
		d3.select("#cross-result-card").classed("invisible", true);*/
		//loading spinner
		d3.select("#src-description-card").classed("visually-hidden", true)
		d3.select("#src-list").classed("visually-hidden", true)
		d3.select("#src-result-card")
			.insert("div").classed("d-flex justify-content-center spinner-align-div", true)
			.append("div").classed("spinner-border", true).attr("role", "status")
  			.append("span").classed("visually-hidden", true).text("Loading...");

  		d3.select("#tgt-description-card").classed("visually-hidden", true)
  		d3.select("#tgt-list").classed("visually-hidden", true)
		d3.select("#tgt-result-card")
			.insert("div").classed("d-flex justify-content-center spinner-align-div", true)
			.append("div").classed("spinner-border", true).attr("role", "status")
  			.append("span").classed("visually-hidden", true).text("Loading...");
  		
  		d3.select("#cross-description-card").classed("visually-hidden", true)
  		d3.select("#cross-list").classed("visually-hidden", true)
  		d3.select("#self-sim-card").classed("visually-hidden", true)
		d3.select("#cross-result-card")
			.insert("div").classed("d-flex justify-content-center spinner-align-div", true)
			.append("div").classed("spinner-border", true).attr("role", "status")
  			.append("span").classed("visually-hidden", true).text("Loading...");

  		//show loading
  		//show_lists(); 
		d3.select("#src-result-card").classed("invisible", false);
		d3.select("#tgt-result-card").classed("invisible", false);
		d3.select("#cross-result-card").classed("invisible", false);

		d3.json("/query-ctx/" + src + "/" + tgt + "/" + word)
			.then(function(response){
				console.log(response);

				d3.selectAll(".spinner-align-div").remove();

				/* *_[s] most similar to word_[s] */
				d3.select("#word-in-src").text(response["word"]);
				d3.select("#src-community").text(srcName);

				if(response['src_sim'] == null){
					d3.select("#word-not-found-in-src").text(response["word"]);
					d3.select("#not-found-in-src-community").text(srcName);
					d3.select("#src-not-found-card").classed("visually-hidden", false);
				} else {
					var srcLi = d3.select("#src-list").selectAll("li")
						.data(response['src_sim'])
						.enter()
						.append("li")
						.classed("list-group-item m-1", true);
						/*.style("background-color", function(d){ 
							var sim = +d['sim'];
							var colorStr = d3.interpolateReds(sim); 
							return colorStr.replace("rgb", "rgba").replace(")", ", " + (sim*0.8) + ")");
						});*/

					srcLi.append("span")
						.classed("badge text-bg-success", true)
						.text(function(d){ return d['word']; });
					/*
					srcLi.append("span")
						.classed("mx-3 px-2", true)
						.classed("text-bg-light", true)
						.text(function(d){ return d['sim'].toFixed(4); });
					*/
					srcLi.append("a")
						.classed("badge rounded-pill text-bg-secondary", true)
						.attr("data-bs-toggle", "collapse")
						.attr("data-bs-target", (d, i) => "#src-ctx-" + i)
						.attr("aria-expanded", "true")
						.attr("aria-controls", (d, i) => "src-ctx-" + i)
						.style("float", "right")
						.text("Toggle context")
					srcLi.append("div")
						.classed("collapse show", true).attr("id", (d, i) => "src-ctx-" + i)
						.append("ul").selectAll("li")
						.data(function(d){ 
							var d_ctx = d['ctx'];
							d_ctx.forEach(function(ctx){
								ctx['sim_word'] = d['word'];
							});
							return d_ctx;
						})
						.enter()
						.append("li")
						.append("span")
						.html(function(d){
							return build_ctx_item_html(d);
						});
					
					d3.select("#src-description-card").classed("visually-hidden", false)
					d3.select("#src-list").classed("visually-hidden", false)
					d3.select("#src-result-card").classed("invisible", false);	
				}
				

				/* *_[t] most similar to word_[t] */
				d3.select("#word-in-tgt").text(response["word"])
				d3.select("#tgt-community").text(tgtName);
				if(response['tgt_sim'] == null){
					d3.select("#word-not-found-in-tgt").text(response["word"]);
					d3.select("#not-found-in-tgt-community").text(tgtName);
					d3.select("#tgt-not-found-card").classed("visually-hidden", false);
				} else {

					var tgtLi = d3.select("#tgt-list").selectAll("li")
						.data(response['tgt_sim'])
						.enter()
						.append("li")
						.classed("list-group-item m-1", true);
						/*.style("background-color", function(d){ 
							var colorStr = d3.interpolateReds(+d['sim']); 
							return colorStr.replace("rgb", "rgba").replace(")", ", 0.6)");
						});*/
					tgtLi.append("span")
						.classed("badge text-bg-primary", true)
						.text(function(d){ return d['word']; });
					/*
					tgtLi.append("span")
						.classed("mx-3 px-2", true)
						.classed("text-bg-light", true)
						.text(function(d){ return d['sim'].toFixed(4); });
					*/
					tgtLi.append("a")
						.classed("badge rounded-pill text-bg-secondary", true)
						.attr("data-bs-toggle", "collapse")
						.attr("data-bs-target", (d, i) => "#tgt-ctx-" + i)
						.attr("aria-expanded", "true")
						.attr("aria-controls", (d, i) => "tgt-ctx-" + i)
						.style("float", "right")
						.text("Toggle context")
					tgtLi.append("div")
						.classed("collapse show", true).attr("id", (d, i) => "tgt-ctx-" + i)
						.append("ul").selectAll("li")
						.data(function(d){ 
							var d_ctx = d['ctx'];
							d_ctx.forEach(function(ctx){
								ctx['sim_word'] = d['word'];
							});
							return d_ctx;
						})
						.enter()
						.append("li")
						.append("span")
						.html(function(d){
							return build_ctx_item_html(d);
						});

					d3.select("#tgt-description-card").classed("visually-hidden", false);
					d3.select("#tgt-list").classed("visually-hidden", false);
					d3.select("#tgt-result-card").classed("invisible", false);
				}

				/* *_[t] most similar to word_[s] */
				if(response['cross_sim'] == null){
				} else {
					d3.select("#word-in-cross").text(response["word"])
					d3.select("#tgt-community-cross").text(tgtName);

					var xLi = d3.select("#cross-list").selectAll("li")
						.data(response['cross_sim'])
						.enter()
						.append("li")
						.classed("list-group-item m-1", true);
						/*.style("background-color", function(d){ 
							var colorStr = d3.interpolateReds(+d['sim']); 
							return colorStr.replace("rgb", "rgba").replace(")", ", 0.6)");
						});*/
					xLi.append("span")
						.classed("badge text-bg-primary", true)
						.text(function(d){ return d['word']; });
					/*
					xLi.append("span")
						.classed("mx-3 px-2", true)
						.classed("text-bg-light", true)
						.text(function(d){ return d['sim'].toFixed(4); });
					*/
					xLi.append("a")
						.classed("badge rounded-pill text-bg-secondary", true)
						.attr("data-bs-toggle", "collapse")
						.attr("data-bs-target", (d, i) => "#cross-ctx-" + i)
						.attr("aria-expanded", "true")
						.attr("aria-controls", (d, i) => "cross-ctx-" + i)
						.style("float", "right")
						.text("Toggle context")
					xLi.append("div")
						.classed("collapse show", true).attr("id", (d, i) => "cross-ctx-" + i)
						.append("ul").selectAll("li")
						.data(function(d){ 
							var d_ctx = d['ctx'];
							d_ctx.forEach(function(ctx){
								ctx['sim_word'] = d['word'];
							});
							return d_ctx;
						})
						.enter()
						.append("li")
						.append("span")
						.html(function(d){
							return build_ctx_item_html(d);
						});
					d3.select("#cross-description-card").classed("visually-hidden", false);
					d3.select("#cross-list").classed("visually-hidden", false);
					d3.select("#cross-result-card").classed("invisible", false);
				}

				/* self similarity & rank */
				/*d3.select("#self-src-word-span").text(response["word"])
				d3.select("#self-tgt-word-span").text(response["word"])
				d3.select("#self-rank-span").text((+response["self_rank"])+1)
				d3.select("#self-sim-span").text(response["self_sim"].toFixed(4))*/

				/*** show all lists ***/
				//show_lists();
			});
	});

function reset_result_panels() {
	d3.select("#src-list").selectAll("*").remove();
	d3.select("#tgt-list").selectAll("*").remove();
	d3.select("#cross-list").selectAll("*").remove();

	d3.select("#src-not-found-card").classed("visually-hidden", true);
	d3.select("#tgt-not-found-card").classed("visually-hidden", true);

	d3.selectAll(".spinner-align-div").remove();
}

function show_lists() {
	d3.select("#src-description-card").classed("visually-hidden", false)
	d3.select("#src-list").classed("visually-hidden", false)
	d3.select("#src-result-card").classed("invisible", false);
	
	d3.select("#tgt-description-card").classed("visually-hidden", false)
	d3.select("#tgt-list").classed("visually-hidden", false)
	d3.select("#tgt-result-card").classed("invisible", false);
	
	d3.select("#cross-description-card").classed("visually-hidden", false)
	d3.select("#cross-list").classed("visually-hidden", false)
	d3.select("#cross-result-card").classed("invisible", false);

	d3.select("#self-sim-card").classed("visually-hidden", false)
}


function build_ctx_item_html(d) {
	var w = d['sim_word'];
	if(w.includes("__"))
		w = w.split("__").join(" ");
	var first_author = JSON.parse(d['authors_json'])[0];

	if(first_author != undefined)
  {
    if('last' in first_author)
      first_author_last_name = first_author['last'];
    else
      first_author_last_name = first_author['name'].split(' ').slice(-1);
  } else {
    first_author_last_name = '';
  }

	//var paper_info_html = '[<a target="_blank" href="https://api.semanticscholar.org/CorpusID:' + d['paper_id'] + '">' + first_author_last_name + " et al. " + d['year'] + "</a>, " + d['section'] + "]";
	var paper_info_html = '[<a target="_blank" href="https://api.semanticscholar.org/' + d['paper_id'] + '">' + first_author_last_name + " et al. " + d['year'] + "</a>, " + d['section'] + "]";

	var w_match_regex = new RegExp("\\b" + w + "\\b", "gi");
	//var w_match_regex = new RegExp(w, "gi");
	/*console.log(w_match_regex);*/
	var sent_html = d['sent'].replace(w_match_regex, function(match){ /*console.log(match);*/ return "<mark>" + match + "</mark>"; });
  	var return_statement = paper_info_html + "&nbsp;" + sent_html;	

	return return_statement
}