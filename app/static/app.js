/* Lead management console.
 *
 * A thin client over the existing API. It holds no business logic: filtering, scoring,
 * extraction and pagination all happen server-side, and this file only renders what comes
 * back. If something here needs to know a threshold or a taxonomy, that is a sign it should
 * have been asked of the API instead.
 *
 * Nodes are built with `el()` rather than innerHTML, so CRM text is always inserted as text
 * and can never be interpreted as markup.
 */
(function () {
  "use strict";

  var PAGE_SIZE = 25;
  var DEDUPE_LIMIT = 8;

  var state = {
    offset: 0,
    total: 0,
    filters: { q: "", status: "", owner: "", country: "" },
  };

  // ------------------------------------------------------------------ utils

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (key) {
        if (key === "class") node.className = attrs[key];
        else if (key === "text") node.textContent = attrs[key];
        else if (key === "html") throw new Error("raw html is not allowed here");
        else if (key.indexOf("on") === 0) node.addEventListener(key.slice(2), attrs[key]);
        else if (attrs[key] !== null && attrs[key] !== undefined) {
          node.setAttribute(key, attrs[key]);
        }
      });
    }
    (children || []).forEach(function (child) {
      if (child === null || child === undefined) return;
      node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
    });
    return node;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function replace(node, child) {
    clear(node);
    node.appendChild(child);
  }

  function stateMessage(message, variant) {
    var cls = "state state--inline";
    if (variant === "error") cls += " state--error";
    return el("p", { class: cls, text: message });
  }

  /* Requests return parsed JSON or throw an Error carrying a message fit to show a user.
   * Validation failures come back from FastAPI with a `detail` field, which is more useful
   * than a status code, so it is preferred when present. */
  function api(path, options) {
    return fetch(path, options)
      .then(function (response) {
        return response
          .json()
          .catch(function () {
            return null;
          })
          .then(function (body) {
            if (response.ok) return body;
            var detail = body && body.detail;
            if (Array.isArray(detail) && detail.length && detail[0].msg) detail = detail[0].msg;
            throw new Error(
              typeof detail === "string" ? detail : "Request failed (" + response.status + ")"
            );
          });
      })
      .catch(function (error) {
        if (error instanceof TypeError) throw new Error("Could not reach the API.");
        throw error;
      });
  }

  function formatNumber(value) {
    return typeof value === "number" ? value.toLocaleString() : String(value);
  }

  function bandClass(confidence) {
    if (confidence === "high") return "tag tag--high";
    if (confidence === "medium") return "tag tag--medium";
    return "tag";
  }

  // -------------------------------------------------------------- dashboard

  function renderKpis(data) {
    var container = document.getElementById("kpis");
    clear(container);

    [
      { label: "Total leads", value: formatNumber(data.total_leads), hint: null },
      {
        label: "Statuses in use",
        value: formatNumber(Object.keys(data.by_status).length),
        hint: "Distinct pipeline stages",
      },
      {
        label: "Needs source review",
        value: formatNumber(data.needs_source_review),
        hint: "Source could not be established from the note",
      },
    ].forEach(function (item) {
      container.appendChild(
        el("div", { class: "kpi" }, [
          el("p", { class: "kpi__label", text: item.label }),
          el("div", { class: "kpi__value", text: item.value }),
          item.hint ? el("p", { class: "kpi__hint", text: item.hint }) : null,
        ])
      );
    });
  }

  /* A CSS bar chart. Deliberately not a charting library: two categorical breakdowns do not
   * justify the dependency, and horizontal bars keep long labels like "Marketing Qualified
   * Lead" readable without rotation. */
  function renderBarChart(target, counts, emptyMessage) {
    var node = document.getElementById(target);
    var entries = Object.keys(counts).map(function (key) {
      return { label: key, value: counts[key] };
    });

    if (!entries.length) {
      replace(node, stateMessage(emptyMessage));
      return;
    }

    entries.sort(function (a, b) {
      return b.value - a.value;
    });
    var max = entries[0].value || 1;
    var total = entries.reduce(function (sum, entry) {
      return sum + entry.value;
    }, 0);

    var list = el("div", { class: "bars", role: "list" });
    entries.forEach(function (entry) {
      var share = total ? Math.round((entry.value / total) * 100) : 0;
      list.appendChild(
        el(
          "div",
          {
            class: "bar",
            role: "listitem",
            "aria-label": entry.label + ": " + entry.value + " leads (" + share + "%)",
          },
          [
            el("span", { class: "bar__label", title: entry.label, text: entry.label }),
            el("div", { class: "bar__track" }, [
              el("div", {
                class: "bar__fill",
                style: "width: " + (entry.value / max) * 100 + "%",
              }),
            ]),
            el("span", { class: "bar__value", text: formatNumber(entry.value) }),
          ]
        )
      );
    });
    replace(node, list);
  }

  /* The status filter is populated from the dashboard's own keys rather than a hardcoded
   * list, so the taxonomy lives in exactly one place: the backend. */
  function populateStatusFilter(byStatus) {
    var select = document.getElementById("f-status");
    Object.keys(byStatus)
      .sort()
      .forEach(function (status) {
        select.appendChild(el("option", { value: status, text: status }));
      });
  }

  function loadDashboard() {
    return api("/dashboard")
      .then(function (data) {
        renderKpis(data);
        renderBarChart("chart-status", data.by_status, "No status data.");
        renderBarChart("chart-source", data.by_source_channel, "No source data.");
        populateStatusFilter(data.by_status);
      })
      .catch(function (error) {
        replace(document.getElementById("kpis"), stateMessage(error.message, "error"));
        replace(document.getElementById("chart-status"), stateMessage("Unavailable", "error"));
        replace(document.getElementById("chart-source"), stateMessage("Unavailable", "error"));
      });
  }

  // ----------------------------------------------------------- lead explorer

  function leadRow(lead) {
    var row = el(
      "tr",
      { tabindex: "0", role: "button", "aria-label": "Open " + (lead.display_name || lead.id) },
      [
        el("td", {}, [
          el("div", { class: "cell-name", text: lead.display_name || "(no name)" }),
          el("div", { class: "cell-sub", text: lead.email || "" }),
        ]),
        el("td", { text: lead.company || "—" }),
        el("td", {}, [el("span", { class: "tag", text: lead.status || "Unknown" })]),
        el("td", {}, [
          lead.source_channel
            ? el("span", {
                class: lead.source_needs_review ? "tag tag--review" : "tag tag--accent",
                text: lead.source_channel,
              })
            : document.createTextNode("—"),
        ]),
        el("td", { class: "col-optional", text: lead.owner || "—" }),
        el("td", { class: "col-optional", text: lead.country || "—" }),
      ]
    );

    function open() {
      showLead(lead.id);
    }
    row.addEventListener("click", open);
    row.addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        open();
      }
    });
    return row;
  }

  function renderPager() {
    var status = document.getElementById("pager-status");
    var prev = document.getElementById("prev-page");
    var next = document.getElementById("next-page");

    if (!state.total) {
      status.textContent = "";
      prev.disabled = true;
      next.disabled = true;
      return;
    }

    var first = state.offset + 1;
    var last = Math.min(state.offset + PAGE_SIZE, state.total);
    status.textContent = first + "–" + last + " of " + formatNumber(state.total);
    prev.disabled = state.offset === 0;
    next.disabled = last >= state.total;
  }

  function loadLeads() {
    var body = document.getElementById("lead-rows");
    var status = document.getElementById("lead-state");
    replace(status, stateMessage("Loading leads…"));

    var params = new URLSearchParams();
    Object.keys(state.filters).forEach(function (key) {
      if (state.filters[key]) params.set(key, state.filters[key]);
    });
    params.set("limit", String(PAGE_SIZE));
    params.set("offset", String(state.offset));

    return api("/leads?" + params.toString())
      .then(function (data) {
        clear(body);
        state.total = data.total;

        if (!data.items.length) {
          replace(status, stateMessage("No leads match these filters."));
        } else {
          clear(status);
          data.items.forEach(function (lead) {
            body.appendChild(leadRow(lead));
          });
        }
        renderPager();
      })
      .catch(function (error) {
        clear(body);
        state.total = 0;
        replace(status, stateMessage(error.message, "error"));
        renderPager();
      });
  }

  // ------------------------------------------------------------ lead detail

  function definitionList(pairs) {
    var list = el("dl", { class: "detail-list" });
    pairs.forEach(function (pair) {
      if (pair[1] === null || pair[1] === undefined || pair[1] === "") return;
      list.appendChild(el("dt", { text: pair[0] }));
      list.appendChild(el("dd", { text: String(pair[1]) }));
    });
    return list;
  }

  function renderLeadDetail(lead) {
    var body = document.getElementById("modal-body");
    clear(body);

    document.getElementById("modal-title").textContent = lead.display_name || "(no name)";
    document.getElementById("modal-sub").textContent = [lead.job_title, lead.company]
      .filter(Boolean)
      .join(" · ");

    body.appendChild(el("p", { class: "section-label", text: "Contact" }));
    body.appendChild(
      definitionList([
        ["Email", lead.email],
        ["Phone", lead.phone],
        ["Country", lead.country],
      ])
    );

    body.appendChild(el("p", { class: "section-label", text: "Pipeline" }));
    body.appendChild(
      definitionList([
        ["Status", lead.status || "Unknown"],
        ["Lifecycle stage", lead.lifecycle_stage],
        ["Owner", lead.owner],
        ["Lead score", lead.lead_score],
        ["Created", lead.created_at],
        ["Last modified", lead.last_modified_at],
      ])
    );

    body.appendChild(el("p", { class: "section-label", text: "Extracted source" }));
    if (lead.source_channel) {
      var sourceRows = [
        ["Channel", lead.source_channel],
        ["Detail", lead.source_detail || "Not stated in the note"],
        ["Confidence", lead.source_confidence],
        ["Method", lead.source_method],
      ];
      body.appendChild(definitionList(sourceRows));
      if (lead.source_needs_review) {
        body.appendChild(
          el("p", { class: "kpi__hint", text: "Flagged for review: the note did not establish a source." })
        );
      }
    } else {
      body.appendChild(stateMessage("No source has been extracted for this lead."));
    }

    body.appendChild(el("p", { class: "section-label", text: "Notes" }));
    body.appendChild(
      lead.notes
        ? el("p", { class: "notes-block", text: lead.notes })
        : stateMessage("No notes recorded.")
    );

    if (lead.raw_record) {
      body.appendChild(
        el("details", {}, [
          el("summary", { text: "Raw source record" }),
          el("pre", { text: JSON.stringify(lead.raw_record, null, 2) }),
        ])
      );
    }
  }

  function showLead(id) {
    var modal = document.getElementById("lead-modal");
    var body = document.getElementById("modal-body");
    document.getElementById("modal-title").textContent = "Lead " + id;
    document.getElementById("modal-sub").textContent = "";
    replace(body, stateMessage("Loading…"));
    if (!modal.open) modal.showModal();

    api("/leads/" + encodeURIComponent(id))
      .then(renderLeadDetail)
      .catch(function (error) {
        replace(body, stateMessage(error.message, "error"));
      });
  }

  // ----------------------------------------------------------------- dedupe

  function renderGroup(entry, kind) {
    var ids = kind === "group" ? entry.lead_ids : [entry.lead_ids[0], entry.lead_ids[1]];
    var summaries = kind === "group" ? entry.summaries : [entry.summaries[0], entry.summaries[1]];

    var head = el("div", { class: "result__head" }, [
      el("span", { class: bandClass(entry.confidence), text: entry.confidence + " confidence" }),
      el("span", { class: "result__score", text: "score " + entry.score }),
      entry.has_internal_conflict
        ? el("span", { class: "tag tag--review", text: "internal conflict" })
        : null,
      kind === "pair" ? el("span", { class: "tag", text: "needs review" }) : null,
    ]);

    var members = el("div", { class: "result__members" });
    ids.forEach(function (id, index) {
      members.appendChild(
        el("div", { class: "result__member" }, [
          el("span", { class: "result__id", text: id }),
          el("span", { text: summaries[index] || "" }),
        ])
      );
    });

    var reasons = el("ul", { class: "reasons" });
    (entry.reasons || []).forEach(function (reason) {
      reasons.appendChild(el("li", { text: reason }));
    });

    return el("li", { class: "result" }, [
      head,
      members,
      el("p", { class: "reasons__heading", text: "Why these were matched" }),
      reasons,
      entry.adjudication ? el("p", { class: "kpi__hint", text: entry.adjudication }) : null,
    ]);
  }

  function runDedupe() {
    var button = document.getElementById("run-dedupe");
    var output = document.getElementById("dedupe-output");
    button.disabled = true;
    replace(output, stateMessage("Scoring candidate pairs…"));

    api("/leads/dedupe-candidates", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ limit: DEDUPE_LIMIT, include_review: true }),
    })
      .then(function (data) {
        clear(output);

        var stats = data.stats || {};
        if (stats.candidate_pairs !== undefined) {
          output.appendChild(
            el("p", {
              class: "card__note",
              text:
                formatNumber(stats.candidate_pairs) +
                " candidate pairs scored instead of " +
                formatNumber(stats.all_pairs_if_brute_forced) +
                " brute-force comparisons · " +
                formatNumber(stats.groups) +
                " groups, " +
                formatNumber(stats.review_pairs) +
                " needing review",
            })
          );
        }

        if (!data.groups.length && !data.review_pairs.length) {
          output.appendChild(stateMessage("No duplicate candidates were found."));
          return;
        }

        var list = el("ul", { class: "result-list" });
        data.groups.forEach(function (group) {
          list.appendChild(renderGroup(group, "group"));
        });
        data.review_pairs.forEach(function (pair) {
          list.appendChild(renderGroup(pair, "pair"));
        });
        output.appendChild(list);
      })
      .catch(function (error) {
        replace(output, stateMessage(error.message, "error"));
      })
      .then(function () {
        button.disabled = false;
      });
  }

  // ------------------------------------------------------------- extraction

  var SAMPLE_NOTES = [
    ["Event", "Spoke with them at our Mobile World Congress booth, no QR scan logged."],
    ["Search", "Googled us and ended up on the pricing page before booking a demo."],
    ["Ambiguous", "Saw our post about replacing hubspot and commented."],
    ["No signal", "Following up after our earlier conversation, please send more info."],
  ];

  function renderExtraction(result) {
    var output = document.getElementById("extract-output");
    clear(output);

    output.appendChild(
      el("div", { class: "result__head" }, [
        el("span", { class: "tag tag--accent", text: result.channel }),
        result.needs_review ? el("span", { class: "tag tag--review", text: "needs review" }) : null,
      ])
    );

    output.appendChild(
      definitionList([
        ["Detail", result.detail || "Not stated in the note"],
        ["Confidence", result.confidence],
        ["Method", result.method],
        ["Matched text", result.evidence || "—"],
      ])
    );

    /* `method` distinguishes the deterministic path from an escalation. Surfacing it makes
     * the rules-first design visible without the caller having to read the docs. */
    var explanation = null;
    if (result.method && result.method.indexOf("rule:") === 0) {
      explanation = "Resolved deterministically. No model call was made.";
    } else if (result.method === "fallback") {
      explanation =
        "The rules could not resolve this note and no model answer was available, so the " +
        "source is left unknown rather than guessed.";
    } else if (result.method && result.method.indexOf("llm") === 0) {
      explanation = "Escalated to the language model because the rules could not resolve it.";
    }
    if (explanation) {
      output.appendChild(el("p", { class: "kpi__hint", text: explanation }));
    }
  }

  function runExtraction(event) {
    event.preventDefault();
    var output = document.getElementById("extract-output");
    var text = document.getElementById("note-text").value;
    replace(output, stateMessage("Extracting…"));

    api("/source/extract", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text }),
    })
      .then(renderExtraction)
      .catch(function (error) {
        replace(output, stateMessage(error.message, "error"));
      });
  }

  function buildSamples() {
    var container = document.getElementById("samples");
    SAMPLE_NOTES.forEach(function (sample) {
      container.appendChild(
        el("button", {
          type: "button",
          text: sample[0],
          onclick: function () {
            document.getElementById("note-text").value = sample[1];
          },
        })
      );
    });
  }

  // ------------------------------------------------------------------ wire

  function init() {
    document.getElementById("filters").addEventListener("submit", function (event) {
      event.preventDefault();
      state.filters = {
        q: document.getElementById("f-q").value.trim(),
        status: document.getElementById("f-status").value,
        owner: document.getElementById("f-owner").value.trim(),
        country: document.getElementById("f-country").value.trim(),
      };
      state.offset = 0;
      loadLeads();
    });

    document.getElementById("filters").addEventListener("reset", function () {
      // The reset happens after this handler, so defer reading the cleared inputs.
      setTimeout(function () {
        state.filters = { q: "", status: "", owner: "", country: "" };
        state.offset = 0;
        loadLeads();
      }, 0);
    });

    document.getElementById("prev-page").addEventListener("click", function () {
      state.offset = Math.max(0, state.offset - PAGE_SIZE);
      loadLeads();
    });

    document.getElementById("next-page").addEventListener("click", function () {
      state.offset = state.offset + PAGE_SIZE;
      loadLeads();
    });

    var modal = document.getElementById("lead-modal");
    document.getElementById("modal-close").addEventListener("click", function () {
      modal.close();
    });
    modal.addEventListener("click", function (event) {
      if (event.target === modal) modal.close();
    });

    document.getElementById("run-dedupe").addEventListener("click", runDedupe);
    document.getElementById("extract-form").addEventListener("submit", runExtraction);

    buildSamples();
    loadDashboard();
    loadLeads();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
