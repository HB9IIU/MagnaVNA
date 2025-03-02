!/**
 * Highcharts JS v12.1.2 (2024-12-21)
 * @module highcharts/modules/annotations
 * @requires highcharts
 *
 * Annotations module
 *
 * (c) 2009-2024 Torstein Honsi
 *
 * License: www.highcharts.com/license
 */function (t, i) {
    "object" == typeof exports && "object" == typeof module ? module.exports = i(t._Highcharts, t._Highcharts.SeriesRegistry, t._Highcharts.Templating, t._Highcharts.AST) : "function" == typeof define && define.amd ? define("highcharts/modules/annotations", ["highcharts/highcharts"], function (t) {
        return i(t, t.SeriesRegistry, t.Templating, t.AST)
    }) : "object" == typeof exports ? exports["highcharts/modules/annotations"] = i(t._Highcharts, t._Highcharts.SeriesRegistry, t._Highcharts.Templating, t._Highcharts.AST) : t.Highcharts = i(t.Highcharts, t.Highcharts.SeriesRegistry, t.Highcharts.Templating, t.Highcharts.AST)
}("undefined" == typeof window ? this : window, (t, i, e, s) => (() => {
    "use strict";
    var o, n, a, r, h = {
        660: t => {
            t.exports = s
        }, 512: t => {
            t.exports = i
        }, 984: t => {
            t.exports = e
        }, 944: i => {
            i.exports = t
        }
    }, l = {};

    function c(t) {
        var i = l[t];
        if (void 0 !== i) return i.exports;
        var e = l[t] = {exports: {}};
        return h[t](e, e.exports, c), e.exports
    }

    c.n = t => {
        var i = t && t.__esModule ? () => t.default : () => t;
        return c.d(i, {a: i}), i
    }, c.d = (t, i) => {
        for (var e in i) c.o(i, e) && !c.o(t, e) && Object.defineProperty(t, e, {enumerable: !0, get: i[e]})
    }, c.o = (t, i) => Object.prototype.hasOwnProperty.call(t, i);
    var p = {};
    c.d(p, {default: () => en});
    var d = c(944), u = c.n(d);
    let {addEvent: g, erase: m, find: f, fireEvent: x, pick: v, wrap: y} = u();

    function b(t, i) {
        let e = this.initAnnotation(t);
        return this.options.annotations.push(e.options), v(i, !0) && (e.redraw(), e.graphic.attr({opacity: 1})), e
    }

    function A() {
        let t = this;
        t.plotBoxClip = this.renderer.clipRect(this.plotBox), t.controlPointsGroup = t.renderer.g("control-points").attr({zIndex: 99}).clip(t.plotBoxClip).add(), t.options.annotations.forEach((i, e) => {
            if (!t.annotations.some(t => t.options === i)) {
                let s = t.initAnnotation(i);
                t.options.annotations[e] = s.options
            }
        }), t.drawAnnotations(), g(t, "redraw", t.drawAnnotations), g(t, "destroy", function () {
            t.plotBoxClip.destroy(), t.controlPointsGroup.destroy()
        }), g(t, "exportData", function (i) {
            let e = t.annotations,
                s = (this.options.exporting && this.options.exporting.csv || {}).columnHeaderFormatter,
                o = !i.dataRows[1].xValues,
                n = t.options.lang && t.options.lang.exportData && t.options.lang.exportData.annotationHeader,
                a = i.dataRows[0].length,
                r = t.options.exporting && t.options.exporting.csv && t.options.exporting.csv.annotations && t.options.exporting.csv.annotations.itemDelimiter,
                h = t.options.exporting && t.options.exporting.csv && t.options.exporting.csv.annotations && t.options.exporting.csv.annotations.join;
            e.forEach(t => {
                t.options.labelOptions && t.options.labelOptions.includeInDataExport && t.labels.forEach(t => {
                    if (t.options.text) {
                        let e = t.options.text;
                        t.points.forEach(t => {
                            let s = t.x, o = t.series.xAxis ? t.series.xAxis.index : -1, n = !1;
                            if (-1 === o) {
                                let t = i.dataRows[0].length, a = Array(t);
                                for (let i = 0; i < t; ++i) a[i] = "";
                                a.push(e), a.xValues = [], a.xValues[o] = s, i.dataRows.push(a), n = !0
                            }
                            if (n || i.dataRows.forEach(t => {
                                !n && t.xValues && void 0 !== o && s === t.xValues[o] && (h && t.length > a ? t[t.length - 1] += r + e : t.push(e), n = !0)
                            }), !n) {
                                let t = i.dataRows[0].length, n = Array(t);
                                for (let i = 0; i < t; ++i) n[i] = "";
                                n[0] = s, n.push(e), n.xValues = [], void 0 !== o && (n.xValues[o] = s), i.dataRows.push(n)
                            }
                        })
                    }
                })
            });
            let l = 0;
            i.dataRows.forEach(t => {
                l = Math.max(l, t.length)
            });
            let c = l - i.dataRows[0].length;
            for (let t = 0; t < c; t++) {
                let e = function (t) {
                    let i;
                    return s && !1 !== (i = s(t)) ? i : (i = n + " " + t, o) ? {
                        columnTitle: i,
                        topLevelColumnTitle: i
                    } : i
                }(t + 1);
                o ? (i.dataRows[0].push(e.topLevelColumnTitle), i.dataRows[1].push(e.columnTitle)) : i.dataRows[0].push(e)
            }
        })
    }

    function k() {
        this.plotBoxClip.attr(this.plotBox), this.annotations.forEach(t => {
            t.redraw(), t.graphic.animate({opacity: 1}, t.animationConfig)
        })
    }

    function w(t) {
        let i = this.annotations, e = "annotations" === t.coll ? t : f(i, function (i) {
            return i.options.id === t
        });
        e && (x(e, "remove"), m(this.options.annotations, e.options), m(i, e), e.destroy())
    }

    function C() {
        this.annotations = [], this.options.annotations || (this.options.annotations = [])
    }

    function E(t) {
        this.chart.hasDraggedAnnotation || t.apply(this, Array.prototype.slice.call(arguments, 1))
    }

    (o || (o = {})).compose = function (t, i, e) {
        let s = i.prototype;
        if (!s.addAnnotation) {
            let o = e.prototype;
            g(i, "afterInit", C), s.addAnnotation = b, s.callbacks.push(A), s.collectionsWithInit.annotations = [b], s.collectionsWithUpdate.push("annotations"), s.drawAnnotations = k, s.removeAnnotation = w, s.initAnnotation = function (i) {
                let e = new (t.types[i.type] || t)(this, i);
                return this.annotations.push(e), e
            }, y(o, "onContainerMouseDown", E)
        }
    };
    let P = o, {defined: O} = u(), {doc: B, isTouchDevice: M} = u(), {
        addEvent: T,
        fireEvent: N,
        objectEach: D,
        pick: L,
        removeEvent: I
    } = u(), S = class {
        addEvents() {
            let t = this, i = function (i) {
                T(i, M ? "touchstart" : "mousedown", i => {
                    t.onMouseDown(i)
                }, {passive: !1})
            };
            if (i(this.graphic.element), (t.labels || []).forEach(t => {
                t.options.useHTML && t.graphic.text && i(t.graphic.text.element)
            }), D(t.options.events, (i, e) => {
                let s = function (s) {
                    "click" === e && t.cancelClick || i.call(t, t.chart.pointer?.normalize(s), t.target)
                };
                -1 === (t.nonDOMEvents || []).indexOf(e) ? (T(t.graphic.element, e, s, {passive: !1}), t.graphic.div && T(t.graphic.div, e, s, {passive: !1})) : T(t, e, s, {passive: !1})
            }), t.options.draggable && (T(t, "drag", t.onDrag), !t.graphic.renderer.styledMode)) {
                let i = {cursor: {x: "ew-resize", y: "ns-resize", xy: "move"}[t.options.draggable]};
                t.graphic.css(i), (t.labels || []).forEach(t => {
                    t.options.useHTML && t.graphic.text && t.graphic.text.css(i)
                })
            }
            t.isUpdating || N(t, "add")
        }

        destroy() {
            this.removeDocEvents(), I(this), this.hcEvents = null
        }

        mouseMoveToRadians(t, i, e) {
            let s = t.prevChartY - e, o = t.prevChartX - i, n = t.chartY - e, a = t.chartX - i, r;
            return this.chart.inverted && (r = o, o = s, s = r, r = a, a = n, n = r), Math.atan2(n, a) - Math.atan2(s, o)
        }

        mouseMoveToScale(t, i, e) {
            let s = t.prevChartX - i, o = t.prevChartY - e, n = t.chartX - i, a = t.chartY - e, r = (n || 1) / (s || 1),
                h = (a || 1) / (o || 1);
            if (this.chart.inverted) {
                let t = h;
                h = r, r = t
            }
            return {x: r, y: h}
        }

        mouseMoveToTranslation(t) {
            let i = t.chartX - t.prevChartX, e = t.chartY - t.prevChartY, s;
            return this.chart.inverted && (s = e, e = i, i = s), {x: i, y: e}
        }

        onDrag(t) {
            if (this.chart.isInsidePlot(t.chartX - this.chart.plotLeft, t.chartY - this.chart.plotTop, {visiblePlotOnly: !0})) {
                let i = this.mouseMoveToTranslation(t);
                "x" === this.options.draggable && (i.y = 0), "y" === this.options.draggable && (i.x = 0), this.points.length ? this.translate(i.x, i.y) : (this.shapes.forEach(t => t.translate(i.x, i.y)), this.labels.forEach(t => t.translate(i.x, i.y))), this.redraw(!1)
            }
        }

        onMouseDown(t) {
            if (t.preventDefault && t.preventDefault(), 2 === t.button) return;
            let i = this, e = i.chart.pointer, s = t?.sourceCapabilities?.firesTouchEvents || !1,
                o = (t = e?.normalize(t) || t).chartX, n = t.chartY;
            i.cancelClick = !1, i.chart.hasDraggedAnnotation = !0, i.removeDrag = T(B, M || s ? "touchmove" : "mousemove", function (t) {
                i.hasDragged = !0, (t = e?.normalize(t) || t).prevChartX = o, t.prevChartY = n, N(i, "drag", t), o = t.chartX, n = t.chartY
            }, M || s ? {passive: !1} : void 0), i.removeMouseUp = T(B, M || s ? "touchend" : "mouseup", function () {
                let t = L(i.target && i.target.annotation, i.target);
                t && (t.cancelClick = i.hasDragged), i.cancelClick = i.hasDragged, i.chart.hasDraggedAnnotation = !1, i.hasDragged && N(L(t, i), "afterUpdate"), i.hasDragged = !1, i.onMouseUp()
            }, M || s ? {passive: !1} : void 0)
        }

        onMouseUp() {
            this.removeDocEvents()
        }

        removeDocEvents() {
            this.removeDrag && (this.removeDrag = this.removeDrag()), this.removeMouseUp && (this.removeMouseUp = this.removeMouseUp())
        }
    }, {merge: Y, pick: X} = u(), F = class extends S {
        constructor(t, i, e, s) {
            super(), this.nonDOMEvents = ["drag"], this.chart = t, this.target = i, this.options = e, this.index = X(e.index, s)
        }

        destroy() {
            super.destroy(), this.graphic && (this.graphic = this.graphic.destroy()), this.chart = null, this.target = null, this.options = null
        }

        redraw(t) {
            this.graphic[t ? "animate" : "attr"](this.options.positioner.call(this, this.target))
        }

        render() {
            let t = this.chart, i = this.options;
            this.graphic = t.renderer.symbol(i.symbol, 0, 0, i.width, i.height).add(t.controlPointsGroup).css(i.style), this.setVisibility(i.visible), this.addEvents()
        }

        setVisibility(t) {
            this.graphic[t ? "show" : "hide"](), this.options.visible = t
        }

        update(t) {
            let i = this.chart, e = this.target, s = this.index, o = Y(!0, this.options, t);
            this.destroy(), this.constructor(i, e, o, s), this.render(i.controlPointsGroup), this.redraw()
        }
    };
    var R = c(512), U = c.n(R);
    let {series: {prototype: W}} = U(), {defined: H, fireEvent: V} = u();

    class j {
        static fromPoint(t) {
            return new j(t.series.chart, null, {x: t.x, y: t.y, xAxis: t.series.xAxis, yAxis: t.series.yAxis})
        }

        static pointToPixels(t, i) {
            let e = t.series, s = e.chart, o = t.plotX || 0, n = t.plotY || 0, a;
            return s.inverted && (t.mock ? (o = t.plotY, n = t.plotX) : (o = s.plotWidth - (t.plotY || 0), n = s.plotHeight - (t.plotX || 0))), e && !i && (o += (a = e.getPlotBox()).translateX, n += a.translateY), {
                x: o,
                y: n
            }
        }

        static pointToOptions(t) {
            return {x: t.x, y: t.y, xAxis: t.series.xAxis, yAxis: t.series.yAxis}
        }

        constructor(t, i, e) {
            this.mock = !0, this.point = this, this.series = {
                visible: !0,
                chart: t,
                getPlotBox: W.getPlotBox
            }, this.target = i || null, this.options = e, this.applyOptions(this.getOptions())
        }

        applyOptions(t) {
            this.command = t.command, this.setAxis(t, "x"), this.setAxis(t, "y"), this.refresh()
        }

        getOptions() {
            return this.hasDynamicOptions() ? this.options(this.target) : this.options
        }

        hasDynamicOptions() {
            return "function" == typeof this.options
        }

        isInsidePlot() {
            let t = this.plotX, i = this.plotY, e = this.series.xAxis, s = this.series.yAxis,
                o = {x: t, y: i, isInsidePlot: !0, options: {}};
            return e && (o.isInsidePlot = H(t) && t >= 0 && t <= e.len), s && (o.isInsidePlot = o.isInsidePlot && H(i) && i >= 0 && i <= s.len), V(this.series.chart, "afterIsInsidePlot", o), o.isInsidePlot
        }

        refresh() {
            let t = this.series, i = t.xAxis, e = t.yAxis, s = this.getOptions();
            i ? (this.x = s.x, this.plotX = i.toPixels(s.x, !0)) : (this.x = void 0, this.plotX = s.x), e ? (this.y = s.y, this.plotY = e.toPixels(s.y, !0)) : (this.y = null, this.plotY = s.y), this.isInside = this.isInsidePlot()
        }

        refreshOptions() {
            let t = this.series, i = t.xAxis, e = t.yAxis;
            this.x = this.options.x = i ? this.options.x = i.toValue(this.plotX, !0) : this.plotX, this.y = this.options.y = e ? e.toValue(this.plotY, !0) : this.plotY
        }

        rotate(t, i, e) {
            if (!this.hasDynamicOptions()) {
                let s = Math.cos(e), o = Math.sin(e), n = this.plotX - t, a = this.plotY - i;
                this.plotX = n * s - a * o + t, this.plotY = n * o + a * s + i, this.refreshOptions()
            }
        }

        scale(t, i, e, s) {
            if (!this.hasDynamicOptions()) {
                let o = this.plotX * e, n = this.plotY * s;
                this.plotX = (1 - e) * t + o, this.plotY = (1 - s) * i + n, this.refreshOptions()
            }
        }

        setAxis(t, i) {
            let e = i + "Axis", s = t[e], o = this.series.chart;
            this.series[e] = "object" == typeof s ? s : H(s) ? o[e][s] || o.get(s) : null
        }

        toAnchor() {
            let t = [this.plotX, this.plotY, 0, 0];
            return this.series.chart.inverted && (t[0] = this.plotY, t[1] = this.plotX), t
        }

        translate(t, i, e, s) {
            this.hasDynamicOptions() || (this.plotX += e, this.plotY += s, this.refreshOptions())
        }
    }

    !function (t) {
        function i() {
            let t = this.controlPoints, i = this.options.controlPoints || [];
            i.forEach((e, s) => {
                let o = u().merge(this.options.controlPointOptions, e);
                o.index || (o.index = s), i[s] = o, t.push(new F(this.chart, this, o))
            })
        }

        function e(t) {
            let i = t.series.getPlotBox(), e = t.series.chart,
                s = t.mock ? t.toAnchor() : e.tooltip && e.tooltip.getAnchor.call({chart: t.series.chart}, t) || [0, 0, 0, 0],
                o = {
                    x: s[0] + (this.options.x || 0),
                    y: s[1] + (this.options.y || 0),
                    height: s[2] || 0,
                    width: s[3] || 0
                };
            return {
                relativePosition: o,
                absolutePosition: u().merge(o, {
                    x: o.x + (t.mock ? i.translateX : e.plotLeft),
                    y: o.y + (t.mock ? i.translateY : e.plotTop)
                })
            }
        }

        function s() {
            this.controlPoints.forEach(t => t.destroy()), this.chart = null, this.controlPoints = null, this.points = null, this.options = null, this.annotation && (this.annotation = null)
        }

        function o() {
            let t = this.options;
            return t.points || t.point && u().splat(t.point)
        }

        function n() {
            let t, i;
            let e = this.getPointsOptions(), s = this.points, o = e && e.length || 0;
            for (t = 0; t < o; t++) {
                if (!(i = this.point(e[t], s[t]))) {
                    s.length = 0;
                    return
                }
                i.mock && i.refresh(), s[t] = i
            }
            return s
        }

        function a(t, i) {
            if (t && t.series) return t;
            if (!i || null === i.series) {
                if (u().isObject(t)) i = new j(this.chart, this, t); else if (u().isString(t)) i = this.chart.get(t) || null; else if ("function" == typeof t) {
                    let e = t.call(i, this);
                    i = e.series ? e : new j(this.chart, this, t)
                }
            }
            return i
        }

        function r(t) {
            this.controlPoints.forEach(i => i.redraw(t))
        }

        function h() {
            this.controlPoints.forEach(t => t.render())
        }

        function l(t, i, e, s, o) {
            if (this.chart.inverted) {
                let t = i;
                i = e, e = t
            }
            this.points.forEach((n, a) => this.transformPoint(t, i, e, s, o, a), this)
        }

        function c(t, i, e, s, o, n) {
            let a = this.points[n];
            a.mock || (a = this.points[n] = j.fromPoint(a)), a[t](i, e, s, o)
        }

        function p(t, i) {
            this.transform("translate", null, null, t, i)
        }

        function d(t, i, e) {
            this.transformPoint("translate", null, null, t, i, e)
        }

        t.compose = function (t) {
            let g = t.prototype;
            g.addControlPoints || u().merge(!0, g, {
                addControlPoints: i,
                anchor: e,
                destroyControlTarget: s,
                getPointsOptions: o,
                linkPoints: n,
                point: a,
                redrawControlPoints: r,
                renderControlPoints: h,
                transform: l,
                transformPoint: c,
                translate: p,
                translatePoint: d
            })
        }
    }(n || (n = {}));
    let q = n, {merge: z} = u();

    class _ {
        constructor(t, i, e, s) {
            this.annotation = t, this.chart = t.chart, this.collection = "label" === s ? "labels" : "shapes", this.controlPoints = [], this.options = i, this.points = [], this.index = e, this.itemType = s, this.init(t, i, e)
        }

        attr(...t) {
            this.graphic.attr.apply(this.graphic, arguments)
        }

        attrsFromOptions(t) {
            let i, e;
            let s = this.constructor.attrsMap, o = {}, n = this.chart.styledMode;
            for (i in t) e = s[i], void 0 === s[i] || n && -1 !== ["fill", "stroke", "stroke-width"].indexOf(e) || (o[e] = t[i]);
            return o
        }

        destroy() {
            this.graphic && (this.graphic = this.graphic.destroy()), this.tracker && (this.tracker = this.tracker.destroy()), this.destroyControlTarget()
        }

        init(t, i, e) {
            this.annotation = t, this.chart = t.chart, this.options = i, this.points = [], this.controlPoints = [], this.index = e, this.linkPoints(), this.addControlPoints()
        }

        redraw(t) {
            this.redrawControlPoints(t)
        }

        render(t) {
            this.options.className && this.graphic && this.graphic.addClass(this.options.className), this.renderControlPoints()
        }

        rotate(t, i, e) {
            this.transform("rotate", t, i, e)
        }

        scale(t, i, e, s) {
            this.transform("scale", t, i, e, s)
        }

        setControlPointsVisibility(t) {
            this.controlPoints.forEach(i => {
                i.setVisibility(t)
            })
        }

        shouldBeDrawn() {
            return !!this.points.length
        }

        translateShape(t, i, e) {
            let s = this.annotation.chart, o = this.annotation.userOptions, n = s.annotations.indexOf(this.annotation),
                a = s.options.annotations[n];
            this.translatePoint(t, i, 0), e && this.translatePoint(t, i, 1), a[this.collection][this.index].point = this.options.point, o[this.collection][this.index].point = this.options.point
        }

        update(t) {
            let i = this.annotation, e = z(!0, this.options, t), s = this.graphic.parentGroup, o = this.constructor;
            this.destroy(), z(!0, this, new o(i, e, this.index, this.itemType)), this.render(s), this.redraw()
        }
    }

    q.compose(_);
    let G = _, {defaultMarkers: K} = {
            defaultMarkers: {
                arrow: {
                    tagName: "marker",
                    attributes: {id: "arrow", refY: 5, refX: 9, markerWidth: 10, markerHeight: 10},
                    children: [{tagName: "path", attributes: {d: "M 0 0 L 10 5 L 0 10 Z", "stroke-width": 0}}]
                },
                "reverse-arrow": {
                    tagName: "marker",
                    attributes: {id: "reverse-arrow", refY: 5, refX: 1, markerWidth: 10, markerHeight: 10},
                    children: [{tagName: "path", attributes: {d: "M 0 5 L 10 0 L 10 10 Z", "stroke-width": 0}}]
                }
            }
        }, {addEvent: Z, defined: $, extend: J, merge: Q, uniqueKey: tt} = u(), ti = to("marker-end"),
        te = to("marker-start"), ts = "rgba(192,192,192," + (u().svg ? 1e-4 : .002) + ")";

    function to(t) {
        return function (i) {
            this.attr(t, "url(#" + i + ")")
        }
    }

    function tn() {
        this.options.defs = Q(K, this.options.defs || {})
    }

    function ta(t, i) {
        let e = {attributes: {id: t}}, s = {stroke: i.color || "none", fill: i.color || "rgba(0, 0, 0, 0.75)"};
        e.children = i.children && i.children.map(function (t) {
            return Q(s, t)
        });
        let o = Q(!0, {attributes: {markerWidth: 20, markerHeight: 20, refX: 0, refY: 0, orient: "auto"}}, i, e),
            n = this.definition(o);
        return n.id = t, n
    }

    class tr extends G {
        static compose(t, i) {
            let e = i.prototype;
            e.addMarker || (Z(t, "afterGetContainer", tn), e.addMarker = ta)
        }

        constructor(t, i, e) {
            super(t, i, e, "shape"), this.type = "path"
        }

        toD() {
            let t = this.options.d;
            if (t) return "function" == typeof t ? t.call(this) : t;
            let i = this.points, e = i.length, s = [], o = e, n = i[0], a = o && this.anchor(n).absolutePosition, r = 0,
                h;
            if (a) for (s.push(["M", a.x, a.y]); ++r < e && o;) h = (n = i[r]).command || "L", a = this.anchor(n).absolutePosition, "M" === h ? s.push([h, a.x, a.y]) : "L" === h ? s.push([h, a.x, a.y]) : "Z" === h && s.push([h]), o = n.series.visible;
            return o && this.graphic ? this.chart.renderer.crispLine(s, this.graphic.strokeWidth()) : null
        }

        shouldBeDrawn() {
            return super.shouldBeDrawn() || !!this.options.d
        }

        render(t) {
            let i = this.options, e = this.attrsFromOptions(i);
            this.graphic = this.annotation.chart.renderer.path([["M", 0, 0]]).attr(e).add(t), this.tracker = this.annotation.chart.renderer.path([["M", 0, 0]]).addClass("highcharts-tracker-line").attr({zIndex: 2}).add(t), this.annotation.chart.styledMode || this.tracker.attr({
                "stroke-linejoin": "round",
                stroke: ts,
                fill: ts,
                "stroke-width": this.graphic.strokeWidth() + 2 * i.snap
            }), super.render(), J(this.graphic, {markerStartSetter: te, markerEndSetter: ti}), this.setMarkers(this)
        }

        redraw(t) {
            if (this.graphic) {
                let i = this.toD(), e = t ? "animate" : "attr";
                i ? (this.graphic[e]({d: i}), this.tracker[e]({d: i})) : (this.graphic.attr({d: "M 0 -9000000000"}), this.tracker.attr({d: "M 0 -9000000000"})), this.graphic.placed = this.tracker.placed = !!i
            }
            super.redraw(t)
        }

        setMarkers(t) {
            let i = t.options, e = t.chart, s = e.options.defs, o = i.fill, n = $(o) && "none" !== o ? o : i.stroke;
            ["markerStart", "markerEnd"].forEach(function (o) {
                let a, r, h, l;
                let c = i[o];
                if (c) {
                    for (h in s) if ((c === ((a = s[h]).attributes && a.attributes.id) || c === a.id) && "marker" === a.tagName) {
                        r = a;
                        break
                    }
                    r && (l = t[o] = e.renderer.addMarker((i.id || tt()) + "-" + c, Q(r, {color: n})), t.attr(o, l.getAttribute("id")))
                }
            })
        }
    }

    tr.attrsMap = {
        dashStyle: "dashstyle",
        strokeWidth: "stroke-width",
        stroke: "stroke",
        fill: "fill",
        zIndex: "zIndex"
    };
    let {merge: th} = u();

    class tl extends G {
        constructor(t, i, e) {
            super(t, i, e, "shape"), this.type = "rect", this.translate = super.translateShape
        }

        render(t) {
            let i = this.attrsFromOptions(this.options);
            this.graphic = this.annotation.chart.renderer.rect(0, -9e9, 0, 0).attr(i).add(t), super.render()
        }

        redraw(t) {
            if (this.graphic) {
                let i = this.anchor(this.points[0]).absolutePosition;
                i ? this.graphic[t ? "animate" : "attr"]({
                    x: i.x,
                    y: i.y,
                    width: this.options.width,
                    height: this.options.height
                }) : this.attr({x: 0, y: -9e9}), this.graphic.placed = !!i
            }
            super.redraw(t)
        }
    }

    tl.attrsMap = th(tr.attrsMap, {width: "width", height: "height"});
    let {merge: tc} = u();

    class tp extends G {
        constructor(t, i, e) {
            super(t, i, e, "shape"), this.type = "circle", this.translate = super.translateShape
        }

        redraw(t) {
            if (this.graphic) {
                let i = this.anchor(this.points[0]).absolutePosition;
                i ? this.graphic[t ? "animate" : "attr"]({x: i.x, y: i.y, r: this.options.r}) : this.graphic.attr({
                    x: 0,
                    y: -9e9
                }), this.graphic.placed = !!i
            }
            super.redraw.call(this, t)
        }

        render(t) {
            let i = this.attrsFromOptions(this.options);
            this.graphic = this.annotation.chart.renderer.circle(0, -9e9, 0).attr(i).add(t), super.render()
        }

        setRadius(t) {
            this.options.r = t
        }
    }

    tp.attrsMap = tc(tr.attrsMap, {r: "r"});
    let {merge: td, defined: tu} = u();

    class tg extends G {
        constructor(t, i, e) {
            super(t, i, e, "shape"), this.type = "ellipse"
        }

        init(t, i, e) {
            tu(i.yAxis) && i.points.forEach(t => {
                t.yAxis = i.yAxis
            }), tu(i.xAxis) && i.points.forEach(t => {
                t.xAxis = i.xAxis
            }), super.init(t, i, e)
        }

        render(t) {
            this.graphic = this.annotation.chart.renderer.createElement("ellipse").attr(this.attrsFromOptions(this.options)).add(t), super.render()
        }

        translate(t, i) {
            super.translateShape(t, i, !0)
        }

        getDistanceFromLine(t, i, e, s) {
            return Math.abs((i.y - t.y) * e - (i.x - t.x) * s + i.x * t.y - i.y * t.x) / Math.sqrt((i.y - t.y) * (i.y - t.y) + (i.x - t.x) * (i.x - t.x))
        }

        getAttrs(t, i) {
            let e = t.x, s = t.y, o = i.x, n = i.y, a = (e + o) / 2,
                r = Math.sqrt((e - o) * (e - o) / 4 + (s - n) * (s - n) / 4),
                h = 180 * Math.atan((n - s) / (o - e)) / Math.PI;
            return a < e && (h += 180), {cx: a, cy: (s + n) / 2, rx: r, ry: this.getRY(), angle: h}
        }

        getRY() {
            let t = this.getYAxis();
            return tu(t) ? Math.abs(t.toPixels(this.options.ry) - t.toPixels(0)) : this.options.ry
        }

        getYAxis() {
            let t = this.options.yAxis;
            return this.chart.yAxis[t]
        }

        getAbsolutePosition(t) {
            return this.anchor(t).absolutePosition
        }

        redraw(t) {
            if (this.graphic) {
                let i = this.getAbsolutePosition(this.points[0]), e = this.getAbsolutePosition(this.points[1]),
                    s = this.getAttrs(i, e);
                i ? this.graphic[t ? "animate" : "attr"]({
                    cx: s.cx,
                    cy: s.cy,
                    rx: s.rx,
                    ry: s.ry,
                    rotation: s.angle,
                    rotationOriginX: s.cx,
                    rotationOriginY: s.cy
                }) : this.graphic.attr({x: 0, y: -9e9}), this.graphic.placed = !!i
            }
            super.redraw(t)
        }

        setYRadius(t) {
            let i = this.annotation.userOptions.shapes;
            this.options.ry = t, i && i[0] && (i[0].ry = t, i[0].ry = t)
        }
    }

    tg.attrsMap = td(tr.attrsMap, {ry: "ry"});
    var tm = c(984), tf = c.n(tm);
    let {format: tx} = tf(), {extend: tv, getAlignFactor: ty, isNumber: tb, pick: tA} = u();

    function tk(t, i, e, s, o) {
        let n = o && o.anchorX, a = o && o.anchorY, r, h, l = e / 2;
        return tb(n) && tb(a) && (r = [["M", n, a]], (h = i - a) < 0 && (h = -s - h), h < e && (l = n < t + e / 2 ? h : e - h), a > i + s ? r.push(["L", t + l, i + s]) : a < i ? r.push(["L", t + l, i]) : n < t ? r.push(["L", t, i + s / 2]) : n > t + e && r.push(["L", t + e, i + s / 2])), r || []
    }

    class tw extends G {
        static alignedPosition(t, i) {
            return {
                x: Math.round((i.x || 0) + (t.x || 0) + (i.width - (t.width || 0)) * ty(t.align)),
                y: Math.round((i.y || 0) + (t.y || 0) + (i.height - (t.height || 0)) * ty(t.verticalAlign))
            }
        }

        static compose(t) {
            t.prototype.symbols.connector = tk
        }

        static justifiedOptions(t, i, e, s) {
            let o;
            let n = e.align, a = e.verticalAlign, r = i.box ? 0 : i.padding || 0, h = i.getBBox(),
                l = {align: n, verticalAlign: a, x: e.x, y: e.y, width: i.width, height: i.height},
                c = (s.x || 0) - t.plotLeft, p = (s.y || 0) - t.plotTop;
            return (o = c + r) < 0 && ("right" === n ? l.align = "left" : l.x = (l.x || 0) - o), (o = c + h.width - r) > t.plotWidth && ("left" === n ? l.align = "right" : l.x = (l.x || 0) + t.plotWidth - o), (o = p + r) < 0 && ("bottom" === a ? l.verticalAlign = "top" : l.y = (l.y || 0) - o), (o = p + h.height - r) > t.plotHeight && ("top" === a ? l.verticalAlign = "bottom" : l.y = (l.y || 0) + t.plotHeight - o), l
        }

        constructor(t, i, e) {
            super(t, i, e, "label")
        }

        translatePoint(t, i) {
            super.translatePoint(t, i, 0)
        }

        translate(t, i) {
            let e = this.annotation.chart, s = this.annotation.userOptions, o = e.annotations.indexOf(this.annotation),
                n = e.options.annotations[o];
            if (e.inverted) {
                let e = t;
                t = i, i = e
            }
            this.options.x += t, this.options.y += i, n[this.collection][this.index].x = this.options.x, n[this.collection][this.index].y = this.options.y, s[this.collection][this.index].x = this.options.x, s[this.collection][this.index].y = this.options.y
        }

        render(t) {
            let i = this.options, e = this.attrsFromOptions(i), s = i.style;
            this.graphic = this.annotation.chart.renderer.label("", 0, -9999, i.shape, null, null, i.useHTML, null, "annotation-label").attr(e).add(t), this.annotation.chart.styledMode || ("contrast" === s.color && (s.color = this.annotation.chart.renderer.getContrast(tw.shapesWithoutBackground.indexOf(i.shape) > -1 ? "#FFFFFF" : i.backgroundColor)), this.graphic.css(i.style).shadow(i.shadow)), this.graphic.labelrank = i.labelrank, super.render()
        }

        redraw(t) {
            let i = this.options, e = this.text || i.format || i.text, s = this.graphic, o = this.points[0];
            if (!s) {
                this.redraw(t);
                return
            }
            s.attr({text: e ? tx(String(e), o, this.annotation.chart) : i.formatter.call(o, this)});
            let n = this.anchor(o), a = this.position(n);
            a ? (s.alignAttr = a, a.anchorX = n.absolutePosition.x, a.anchorY = n.absolutePosition.y, s[t ? "animate" : "attr"](a)) : s.attr({
                x: 0,
                y: -9999
            }), s.placed = !!a, super.redraw(t)
        }

        anchor(t) {
            let i = super.anchor.apply(this, arguments), e = this.options.x || 0, s = this.options.y || 0;
            return i.absolutePosition.x -= e, i.absolutePosition.y -= s, i.relativePosition.x -= e, i.relativePosition.y -= s, i
        }

        position(t) {
            let i = this.graphic, e = this.annotation.chart, s = e.tooltip, o = this.points[0], n = this.options,
                a = t.absolutePosition, r = t.relativePosition, h, l, c, p,
                d = o.series.visible && j.prototype.isInsidePlot.call(o);
            if (i && d) {
                let {width: t = 0, height: u = 0} = i;
                n.distance && s ? h = s.getPosition.call({
                    chart: e,
                    distance: tA(n.distance, 16),
                    getPlayingField: s.getPlayingField,
                    pointer: s.pointer
                }, t, u, {
                    plotX: r.x,
                    plotY: r.y,
                    negative: o.negative,
                    ttBelow: o.ttBelow,
                    h: r.height || r.width
                }) : n.positioner ? h = n.positioner.call(this) : (l = {
                    x: a.x,
                    y: a.y,
                    width: 0,
                    height: 0
                }, h = tw.alignedPosition(tv(n, {
                    width: t,
                    height: u
                }), l), "justify" === this.options.overflow && (h = tw.alignedPosition(tw.justifiedOptions(e, i, n, h), l))), n.crop && (c = h.x - e.plotLeft, p = h.y - e.plotTop, d = e.isInsidePlot(c, p) && e.isInsidePlot(c + t, p + u))
            }
            return d ? h : null
        }
    }

    tw.attrsMap = {
        backgroundColor: "fill",
        borderColor: "stroke",
        borderWidth: "stroke-width",
        zIndex: "zIndex",
        borderRadius: "r",
        padding: "padding"
    }, tw.shapesWithoutBackground = ["connector"];

    class tC extends G {
        constructor(t, i, e) {
            super(t, i, e, "shape"), this.type = "image", this.translate = super.translateShape
        }

        render(t) {
            let i = this.attrsFromOptions(this.options), e = this.options;
            this.graphic = this.annotation.chart.renderer.image(e.src, 0, -9e9, e.width, e.height).attr(i).add(t), this.graphic.width = e.width, this.graphic.height = e.height, super.render()
        }

        redraw(t) {
            if (this.graphic) {
                let i = this.anchor(this.points[0]), e = tw.prototype.position.call(this, i);
                e ? this.graphic[t ? "animate" : "attr"]({x: e.x, y: e.y}) : this.graphic.attr({
                    x: 0,
                    y: -9e9
                }), this.graphic.placed = !!e
            }
            super.redraw(t)
        }
    }

    tC.attrsMap = {width: "width", height: "height", zIndex: "zIndex"};
    var tE = c(660), tP = c.n(tE);
    let {addEvent: tO, createElement: tB} = u(), tM = class {
        constructor(t, i) {
            this.iconsURL = i, this.container = this.createPopupContainer(t), this.closeButton = this.addCloseButton()
        }

        createPopupContainer(t, i = "highcharts-popup highcharts-no-tooltip") {
            return tB("div", {className: i}, void 0, t)
        }

        addCloseButton(t = "highcharts-popup-close") {
            let i = this, e = this.iconsURL, s = tB("button", {className: t}, void 0, this.container);
            return s.style["background-image"] = "url(" + (e.match(/png|svg|jpeg|jpg|gif/ig) ? e : e + "close.svg") + ")", ["click", "touchstart"].forEach(t => {
                tO(s, t, i.closeButtonEvents.bind(i))
            }), tO(document, "keydown", function (t) {
                "Escape" === t.code && i.closeButtonEvents()
            }), s
        }

        closeButtonEvents() {
            this.closePopup()
        }

        showPopup(t = "highcharts-annotation-toolbar") {
            let i = this.container, e = this.closeButton;
            this.type = void 0, i.innerHTML = tP().emptyHTML, i.className.indexOf(t) >= 0 && (i.classList.remove(t), i.removeAttribute("style")), i.appendChild(e), i.style.display = "block", i.style.height = ""
        }

        closePopup() {
            this.container.style.display = "none"
        }
    }, {doc: tT, isFirefox: tN} = u(), {
        createElement: tD,
        isArray: tL,
        isObject: tI,
        objectEach: tS,
        pick: tY,
        stableSort: tX
    } = u();

    function tF(t, i, e, s, o, n) {
        let a, r;
        if (!i) return;
        let h = this.addInput, l = this.lang;
        tS(s, (s, n) => {
            a = "" !== e ? e + "." + n : n, tI(s) && (!tL(s) || tL(s) && tI(s[0]) ? ((r = l[n] || n).match(/\d/g) || o.push([!0, r, t]), tF.call(this, t, i, a, s, o, !1)) : o.push([this, a, "annotation", t, s]))
        }), n && (tX(o, t => t[1].match(/format/g) ? -1 : 1), tN && o.reverse(), o.forEach(t => {
            !0 === t[0] ? tD("span", {className: "highcharts-annotation-title"}, void 0, t[2]).appendChild(tT.createTextNode(t[1])) : (t[4] = {
                value: t[4][0],
                type: t[4][1]
            }, h.apply(t[0], t.splice(1)))
        }))
    }

    let {doc: tR} = u(), {seriesTypes: tU} = U(), {
        addEvent: tW,
        createElement: tH,
        defined: tV,
        isArray: tj,
        isObject: tq,
        objectEach: tz,
        stableSort: t_
    } = u();
    !function (t) {
        t[t["params.algorithm"] = 0] = "params.algorithm", t[t["params.average"] = 1] = "params.average"
    }(a || (a = {}));
    let tG = {
        "algorithm-pivotpoints": ["standard", "fibonacci", "camarilla"],
        "average-disparityindex": ["sma", "ema", "dema", "tema", "wma"]
    };

    function tK(t) {
        let i = tH("div", {className: "highcharts-popup-lhs-col"}, void 0, t),
            e = tH("div", {className: "highcharts-popup-rhs-col"}, void 0, t);
        return tH("div", {className: "highcharts-popup-rhs-col-wrapper"}, void 0, e), {lhsCol: i, rhsCol: e}
    }

    function tZ(t, i, e, s) {
        let o = i.params || i.options.params;
        s.innerHTML = tP().emptyHTML, tH("h3", {className: "highcharts-indicator-title"}, void 0, s).appendChild(tR.createTextNode(t4(i, e).indicatorFullName)), tH("input", {
            type: "hidden",
            name: "highcharts-type-" + e,
            value: e
        }, void 0, s), t5.call(this, e, "series", t, s, i, i.linkedParent && i.linkedParent.options.id), o.volumeSeriesID && t5.call(this, e, "volume", t, s, i, i.linkedParent && o.volumeSeriesID), tJ.call(this, t, "params", o, e, s)
    }

    function t$(t, i, e, s) {
        function o(i, e) {
            let s = g.parentNode.children[1];
            tZ.call(n, t, i, e, g), s && (s.style.display = "block"), l && i.options && tH("input", {
                type: "hidden",
                name: "highcharts-id-" + e,
                value: i.options.id
            }, void 0, g).setAttribute("highcharts-data-series-id", i.options.id)
        }

        let n = this, a = n.lang, r = i.querySelectorAll(".highcharts-popup-lhs-col")[0],
            h = i.querySelectorAll(".highcharts-popup-rhs-col")[0], l = "edit" === e,
            c = l ? t.series : t.options.plotOptions || {};
        if (!t && c) return;
        let p, d = [];
        l || tj(c) ? tj(c) && (d = t2.call(this, c)) : d = t9.call(this, c, s), t_(d, (t, i) => {
            let e = t.indicatorFullName.toLowerCase(), s = i.indicatorFullName.toLowerCase();
            return e < s ? -1 : e > s ? 1 : 0
        }), r.children[1] && r.children[1].remove();
        let u = tH("ul", {className: "highcharts-indicator-list"}, void 0, r),
            g = h.querySelectorAll(".highcharts-popup-rhs-col-wrapper")[0];
        if (d.forEach(t => {
            let {indicatorFullName: i, indicatorType: e, series: s} = t;
            p = tH("li", {className: "highcharts-indicator-list"}, void 0, u);
            let n = tH("button", {className: "highcharts-indicator-list-item", textContent: i}, void 0, p);
            ["click", "touchstart"].forEach(t => {
                tW(n, t, function () {
                    o(s, e)
                })
            })
        }), d.length > 0) {
            let {series: t, indicatorType: i} = d[0];
            o(t, i)
        } else l || (tP().setElementHTML(g.parentNode.children[0], a.noFilterMatch || ""), g.parentNode.children[1].style.display = "none")
    }

    function tJ(t, i, e, s, o) {
        if (!t) return;
        let n = this.addInput;
        tz(e, (e, r) => {
            let h = i + "." + r;
            if (tV(e) && h) {
                if (tq(e) && (n.call(this, h, s, o, {}), tJ.call(this, t, h, e, s, o)), h in a) {
                    let n = t0.call(this, s, h, o);
                    t1.call(this, t, i, n, s, r, e)
                } else "params.volumeSeriesID" === h || tj(e) || n.call(this, h, s, o, {value: e, type: "number"})
            }
        })
    }

    function tQ(t, i) {
        let e = this, s = i.querySelectorAll(".highcharts-popup-lhs-col")[0], o = this.lang.clearFilter,
            n = tH("div", {className: "highcharts-input-wrapper"}, void 0, s), a = function (i) {
                t$.call(e, t, e.container, "add", i)
            }, r = this.addInput("searchIndicators", "input", n, {
                value: "",
                type: "text",
                htmlFor: "search-indicators",
                labelClassName: "highcharts-input-search-indicators-label"
            }), h = tH("a", {textContent: o}, void 0, n);
        r.classList.add("highcharts-input-search-indicators"), h.classList.add("clear-filter-button"), tW(r, "input", function () {
            a(this.value), this.value.length ? h.style.display = "inline-block" : h.style.display = "none"
        }), ["click", "touchstart"].forEach(t => {
            tW(h, t, function () {
                r.value = "", a(""), h.style.display = "none"
            })
        })
    }

    function t0(t, i, e) {
        let s = i.split("."), o = s[s.length - 1], n = "highcharts-" + i + "-type-" + t, a = this.lang;
        tH("label", {htmlFor: n}, null, e).appendChild(tR.createTextNode(a[o] || i));
        let r = tH("select", {name: n, className: "highcharts-popup-field", id: "highcharts-select-" + i}, null, e);
        return r.setAttribute("id", "highcharts-select-" + i), r
    }

    function t1(t, i, e, s, o, n, a) {
        "series" === i || "volume" === i ? t.series.forEach(t => {
            let s = t.options, o = s.name || s.params ? t.name : s.id || "";
            "highcharts-navigator-series" !== s.id && s.id !== (a && a.options && a.options.id) && (tV(n) || "volume" !== i || "column" !== t.type || (n = s.id), tH("option", {value: s.id}, void 0, e).appendChild(tR.createTextNode(o)))
        }) : s && o && tG[o + "-" + s].forEach(t => {
            tH("option", {value: t}, void 0, e).appendChild(tR.createTextNode(t))
        }), tV(n) && (e.value = n)
    }

    function t9(t, i) {
        let e;
        let s = this.chart && this.chart.options.lang,
            o = s && s.navigation && s.navigation.popup && s.navigation.popup.indicatorAliases, n = [];
        return tz(t, (t, s) => {
            let a = t && t.options;
            if (t.params || a && a.params) {
                let {indicatorFullName: a, indicatorType: r} = t4(t, s);
                if (i) {
                    let s = RegExp(i.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "i"),
                        h = o && o[r] && o[r].join(" ") || "";
                    (a.match(s) || h.match(s)) && (e = {indicatorFullName: a, indicatorType: r, series: t}, n.push(e))
                } else e = {indicatorFullName: a, indicatorType: r, series: t}, n.push(e)
            }
        }), n
    }

    function t2(t) {
        let i = [];
        return t.forEach(t => {
            t.is("sma") && i.push({indicatorFullName: t.name, indicatorType: t.type, series: t})
        }), i
    }

    function t4(t, i) {
        let e = t.options, s = tU[i] && tU[i].prototype.nameBase || i.toUpperCase(), o = i;
        return e && e.type && (o = t.options.type, s = t.name), {indicatorFullName: s, indicatorType: o}
    }

    function t5(t, i, e, s, o, n) {
        if (!e) return;
        let a = t0.call(this, t, i, s);
        t1.call(this, e, i, a, void 0, void 0, void 0, o), tV(n) && (a.value = n)
    }

    let {doc: t6} = u(), {addEvent: t7, createElement: t3} = u();

    function t8() {
        return t3("div", {className: "highcharts-tab-item-content highcharts-no-mousewheel"}, void 0, this.container)
    }

    function it(t, i) {
        let e = this.container, s = this.lang, o = "highcharts-tab-item";
        0 === i && (o += " highcharts-tab-disabled");
        let n = t3("button", {className: o}, void 0, e);
        return n.appendChild(t6.createTextNode(s[t + "Button"] || t)), n.setAttribute("highcharts-data-tab-type", t), n
    }

    function ii() {
        let t = this.container, i = t.querySelectorAll(".highcharts-tab-item"),
            e = t.querySelectorAll(".highcharts-tab-item-content");
        for (let t = 0; t < i.length; t++) i[t].classList.remove("highcharts-tab-item-active"), e[t].classList.remove("highcharts-tab-item-show")
    }

    function ie(t, i) {
        let e = this.container.querySelectorAll(".highcharts-tab-item-content");
        t.className += " highcharts-tab-item-active", e[i].className += " highcharts-tab-item-show"
    }

    function is(t) {
        let i = this;
        this.container.querySelectorAll(".highcharts-tab-item").forEach((e, s) => {
            (0 !== t || "edit" !== e.getAttribute("highcharts-data-tab-type")) && ["click", "touchstart"].forEach(t => {
                t7(e, t, function () {
                    ii.call(i), ie.call(i, this, s)
                })
            })
        })
    }

    let {doc: io} = u(), {getOptions: ia} = u(), {
        addEvent: ir,
        createElement: ih,
        extend: il,
        fireEvent: ic,
        pick: ip
    } = u();

    class id extends tM {
        constructor(t, i, e) {
            super(t, i), this.chart = e, this.lang = (ia().lang.navigation || {}).popup || {}, ir(this.container, "mousedown", () => {
                let t = e && e.navigationBindings && e.navigationBindings.activeAnnotation;
                if (t) {
                    t.cancelClick = !0;
                    let i = ir(io, "click", () => {
                        setTimeout(() => {
                            t.cancelClick = !1
                        }, 0), i()
                    })
                }
            })
        }

        addInput(t, i, e, s) {
            let o = t.split("."), n = o[o.length - 1], a = this.lang, r = "highcharts-" + i + "-" + ip(s.htmlFor, n);
            n.match(/^\d+$/) || ih("label", {
                htmlFor: r,
                className: s.labelClassName
            }, void 0, e).appendChild(io.createTextNode(a[n] || n));
            let h = ih("input", {
                name: r,
                value: s.value,
                type: s.type,
                className: "highcharts-popup-field"
            }, void 0, e);
            return h.setAttribute("highcharts-data-name", t), h
        }

        closeButtonEvents() {
            if (this.chart) {
                let t = this.chart.navigationBindings;
                ic(t, "closePopup"), t && t.selectedButtonElement && ic(t, "deselectButton", {button: t.selectedButtonElement})
            } else super.closeButtonEvents()
        }

        addButton(t, i, e, s, o) {
            let n = ih("button", void 0, void 0, t);
            return n.appendChild(io.createTextNode(i)), o && ["click", "touchstart"].forEach(t => {
                ir(n, t, () => (this.closePopup(), o(function (t, i) {
                    let e = Array.prototype.slice.call(t.querySelectorAll("input")),
                        s = Array.prototype.slice.call(t.querySelectorAll("select")),
                        o = t.querySelectorAll("#highcharts-select-series > option:checked")[0],
                        n = t.querySelectorAll("#highcharts-select-volume > option:checked")[0],
                        a = {actionType: i, linkedTo: o && o.getAttribute("value") || "", fields: {}};
                    return e.forEach(t => {
                        let i = t.getAttribute("highcharts-data-name");
                        t.getAttribute("highcharts-data-series-id") ? a.seriesId = t.value : i ? a.fields[i] = t.value : a.type = t.value
                    }), s.forEach(t => {
                        let i = t.id;
                        if ("highcharts-select-series" !== i && "highcharts-select-volume" !== i) {
                            let e = i.split("highcharts-select-")[1];
                            a.fields[e] = t.value
                        }
                    }), n && (a.fields["params.volumeSeriesID"] = n.getAttribute("value") || ""), a
                }(s, e))))
            }), n
        }

        showForm(t, i, e, s) {
            i && (this.showPopup(), "indicators" === t && this.indicators.addForm.call(this, i, e, s), "annotation-toolbar" === t && this.annotations.addToolbar.call(this, i, e, s), "annotation-edit" === t && this.annotations.addForm.call(this, i, e, s), "flag" === t && this.annotations.addForm.call(this, i, e, s, !0), this.type = t, this.container.style.height = this.container.offsetHeight + "px")
        }
    }

    il(id.prototype, {
        annotations: {
            addForm: function (t, i, e, s) {
                if (!t) return;
                let o = this.container, n = this.lang,
                    a = tD("h2", {className: "highcharts-popup-main-title"}, void 0, o);
                a.appendChild(tT.createTextNode(n[i.langKey] || i.langKey || "")), a = tD("div", {className: "highcharts-popup-lhs-col highcharts-popup-lhs-full"}, void 0, o);
                let r = tD("div", {className: "highcharts-popup-bottom-row"}, void 0, o);
                tF.call(this, a, t, "", i, [], !0), this.addButton(r, s ? n.addButton || "Add" : n.saveButton || "Save", s ? "add" : "save", o, e)
            }, addToolbar: function (t, i, e) {
                let s = this.lang, o = this.container, n = this.showForm, a = "highcharts-annotation-toolbar";
                -1 === o.className.indexOf(a) && (o.className += " " + a + " highcharts-no-mousewheel"), t && (o.style.top = t.plotTop + 10 + "px");
                let r = tD("p", {className: "highcharts-annotation-label"}, void 0, o);
                r.setAttribute("aria-label", "Annotation type"), r.appendChild(tT.createTextNode(tY(s[i.langKey] || i.langKey, i.shapes && i.shapes[0].type, "")));
                let h = this.addButton(o, s.editButton || "Edit", "edit", o, () => {
                    n.call(this, "annotation-edit", t, i, e)
                });
                h.className += " highcharts-annotation-edit-button", h.style["background-image"] = "url(" + this.iconsURL + "edit.svg)", h = this.addButton(o, s.removeButton || "Remove", "remove", o, e), h.className += " highcharts-annotation-remove-button", h.style["background-image"] = "url(" + this.iconsURL + "destroy.svg)"
            }
        }, indicators: {
            addForm: function (t, i, e) {
                let s;
                let o = this.lang;
                if (!t) return;
                this.tabs.init.call(this, t);
                let n = this.container.querySelectorAll(".highcharts-tab-item-content");
                tK(n[0]), tQ.call(this, t, n[0]), t$.call(this, t, n[0], "add"), s = n[0].querySelectorAll(".highcharts-popup-rhs-col")[0], this.addButton(s, o.addButton || "add", "add", s, e), tK(n[1]), t$.call(this, t, n[1], "edit"), s = n[1].querySelectorAll(".highcharts-popup-rhs-col")[0], this.addButton(s, o.saveButton || "save", "edit", s, e), this.addButton(s, o.removeButton || "remove", "remove", s, e)
            }, getAmount: function () {
                let t = 0;
                return this.series.forEach(i => {
                    (i.params || i.options.params) && t++
                }), t
            }
        }, tabs: {
            init: function (t) {
                if (!t) return;
                let i = this.indicators.getAmount.call(t), e = it.call(this, "add");
                it.call(this, "edit", i), t8.call(this), t8.call(this), is.call(this, i), ie.call(this, e, 0)
            }
        }
    });
    let {composed: iu} = u(), {addEvent: ig, pushUnique: im, wrap: ix} = u();

    function iv() {
        this.popup && this.popup.closePopup()
    }

    function iy(t) {
        this.popup || (this.popup = new id(this.chart.container, this.chart.options.navigation.iconsURL || this.chart.options.stockTools && this.chart.options.stockTools.gui.iconsURL || "https://code.highcharts.com/12.1.2/gfx/stock-icons/", this.chart)), this.popup.showForm(t.formType, this.chart, t.options, t.onSubmit)
    }

    function ib(t, i) {
        this.inClass(i.target, "highcharts-popup") || t.apply(this, Array.prototype.slice.call(arguments, 1))
    }

    let iA = {
        compose: function (t, i) {
            im(iu, "Popup") && (ig(t, "closePopup", iv), ig(t, "showPopup", iy), ix(i.prototype, "onContainerMouseDown", ib))
        }
    }, {getDeferredAnimation: ik} = u(), {
        destroyObjectProperties: iw,
        erase: iC,
        fireEvent: iE,
        merge: iP,
        pick: iO,
        splat: iB
    } = u();

    function iM(t, i) {
        let e = {};
        return ["labels", "shapes"].forEach(s => {
            let o = t[s], n = i[s];
            o && (n ? e[s] = iB(n).map((t, i) => iP(o[i], t)) : e[s] = t[s])
        }), e
    }

    class iT extends S {
        static compose(t, i, e, s) {
            P.compose(iT, t, e), tw.compose(s), tr.compose(t, s), i.compose(iT, t), iA.compose(i, e)
        }

        constructor(t, i) {
            super(), this.coll = "annotations", this.chart = t, this.points = [], this.controlPoints = [], this.coll = "annotations", this.index = -1, this.labels = [], this.shapes = [], this.options = iP(this.defaultOptions, i), this.userOptions = i;
            let e = iM(this.options, i);
            this.options.labels = e.labels, this.options.shapes = e.shapes, this.init(t, this.options)
        }

        addClipPaths() {
            this.setClipAxes(), this.clipXAxis && this.clipYAxis && this.options.crop && (this.clipRect = this.chart.renderer.clipRect(this.getClipBox()))
        }

        addLabels() {
            let t = this.options.labels || [];
            t.forEach((i, e) => {
                let s = this.initLabel(i, e);
                iP(!0, t[e], s.options)
            })
        }

        addShapes() {
            let t = this.options.shapes || [];
            t.forEach((i, e) => {
                let s = this.initShape(i, e);
                iP(!0, t[e], s.options)
            })
        }

        destroy() {
            let t = this.chart, i = function (t) {
                t.destroy()
            };
            this.labels.forEach(i), this.shapes.forEach(i), this.clipXAxis = null, this.clipYAxis = null, iC(t.labelCollectors, this.labelCollector), super.destroy(), this.destroyControlTarget(), iw(this, t)
        }

        destroyItem(t) {
            iC(this[t.itemType + "s"], t), t.destroy()
        }

        getClipBox() {
            if (this.clipXAxis && this.clipYAxis) return {
                x: this.clipXAxis.left,
                y: this.clipYAxis.top,
                width: this.clipXAxis.width,
                height: this.clipYAxis.height
            }
        }

        initProperties(t, i) {
            this.setOptions(i);
            let e = iM(this.options, i);
            this.options.labels = e.labels, this.options.shapes = e.shapes, this.chart = t, this.points = [], this.controlPoints = [], this.coll = "annotations", this.userOptions = i, this.labels = [], this.shapes = []
        }

        init(t, i, e = this.index) {
            let s = this.chart, o = this.options.animation;
            this.index = e, this.linkPoints(), this.addControlPoints(), this.addShapes(), this.addLabels(), this.setLabelCollector(), this.animationConfig = ik(s, o)
        }

        initLabel(t, i) {
            let e = new tw(this, iP(this.options.labelOptions, {controlPointOptions: this.options.controlPointOptions}, t), i);
            return e.itemType = "label", this.labels.push(e), e
        }

        initShape(t, i) {
            let e = iP(this.options.shapeOptions, {controlPointOptions: this.options.controlPointOptions}, t),
                s = new iT.shapesMap[e.type](this, e, i);
            return s.itemType = "shape", this.shapes.push(s), s
        }

        redraw(t) {
            this.linkPoints(), this.graphic || this.render(), this.clipRect && this.clipRect.animate(this.getClipBox()), this.redrawItems(this.shapes, t), this.redrawItems(this.labels, t), this.redrawControlPoints(t)
        }

        redrawItem(t, i) {
            t.linkPoints(), t.shouldBeDrawn() ? (t.graphic || this.renderItem(t), t.redraw(iO(i, !0) && t.graphic.placed), t.points.length && function (t) {
                let i = t.graphic, e = t.points.some(t => !1 !== t.series.visible && !1 !== t.visible);
                i && (e ? "hidden" === i.visibility && i.show() : i.hide())
            }(t)) : this.destroyItem(t)
        }

        redrawItems(t, i) {
            let e = t.length;
            for (; e--;) this.redrawItem(t[e], i)
        }

        remove() {
            return this.chart.removeAnnotation(this)
        }

        render() {
            let t = this.chart.renderer;
            this.graphic = t.g("annotation").attr({
                opacity: 0,
                zIndex: this.options.zIndex,
                visibility: this.options.visible ? "inherit" : "hidden"
            }).add(), this.shapesGroup = t.g("annotation-shapes").add(this.graphic), this.options.crop && this.shapesGroup.clip(this.chart.plotBoxClip), this.labelsGroup = t.g("annotation-labels").attr({
                translateX: 0,
                translateY: 0
            }).add(this.graphic), this.addClipPaths(), this.clipRect && this.graphic.clip(this.clipRect), this.renderItems(this.shapes), this.renderItems(this.labels), this.addEvents(), this.renderControlPoints()
        }

        renderItem(t) {
            t.render("label" === t.itemType ? this.labelsGroup : this.shapesGroup)
        }

        renderItems(t) {
            let i = t.length;
            for (; i--;) this.renderItem(t[i])
        }

        setClipAxes() {
            let t = this.chart.xAxis, i = this.chart.yAxis,
                e = (this.options.labels || []).concat(this.options.shapes || []).reduce((e, s) => {
                    let o = s && (s.point || s.points && s.points[0]);
                    return [t[o && o.xAxis] || e[0], i[o && o.yAxis] || e[1]]
                }, []);
            this.clipXAxis = e[0], this.clipYAxis = e[1]
        }

        setControlPointsVisibility(t) {
            let i = function (i) {
                i.setControlPointsVisibility(t)
            };
            this.controlPoints.forEach(i => {
                i.setVisibility(t)
            }), this.shapes.forEach(i), this.labels.forEach(i)
        }

        setLabelCollector() {
            let t = this;
            t.labelCollector = function () {
                return t.labels.reduce(function (t, i) {
                    return i.options.allowOverlap || t.push(i.graphic), t
                }, [])
            }, t.chart.labelCollectors.push(t.labelCollector)
        }

        setOptions(t) {
            this.options = iP(this.defaultOptions, t)
        }

        setVisibility(t) {
            let i = this.options, e = this.chart.navigationBindings, s = iO(t, !i.visible);
            if (this.graphic.attr("visibility", s ? "inherit" : "hidden"), !s) {
                let t = function (t) {
                    t.setControlPointsVisibility(s)
                };
                this.shapes.forEach(t), this.labels.forEach(t), e.activeAnnotation === this && e.popup && "annotation-toolbar" === e.popup.type && iE(e, "closePopup")
            }
            i.visible = s
        }

        update(t, i) {
            let e = this.chart, s = iM(this.userOptions, t), o = e.annotations.indexOf(this),
                n = iP(!0, this.userOptions, t);
            n.labels = s.labels, n.shapes = s.shapes, this.destroy(), this.initProperties(e, n), this.init(e, n), e.options.annotations[o] = this.options, this.isUpdating = !0, iO(i, !0) && e.drawAnnotations(), iE(this, "afterUpdate"), this.isUpdating = !1
        }
    }

    iT.ControlPoint = F, iT.MockPoint = j, iT.shapesMap = {
        rect: tl,
        circle: tp,
        ellipse: tg,
        path: tr,
        image: tC
    }, iT.types = {}, iT.prototype.defaultOptions = {
        visible: !0,
        animation: {},
        crop: !0,
        draggable: "xy",
        labelOptions: {
            align: "center",
            allowOverlap: !1,
            backgroundColor: "rgba(0, 0, 0, 0.75)",
            borderColor: "#000000",
            borderRadius: 3,
            borderWidth: 1,
            className: "highcharts-no-tooltip",
            crop: !1,
            formatter: function () {
                return O(this.y) ? "" + this.y : "Annotation label"
            },
            includeInDataExport: !0,
            overflow: "justify",
            padding: 5,
            shadow: !1,
            shape: "callout",
            style: {fontSize: "0.7em", fontWeight: "normal", color: "contrast"},
            useHTML: !1,
            verticalAlign: "bottom",
            x: 0,
            y: -16
        },
        shapeOptions: {stroke: "rgba(0, 0, 0, 0.75)", strokeWidth: 1, fill: "rgba(0, 0, 0, 0.75)", r: 0, snap: 2},
        controlPointOptions: {
            events: {},
            style: {cursor: "pointer", fill: "#ffffff", stroke: "#000000", "stroke-width": 2},
            height: 10,
            symbol: "circle",
            visible: !1,
            width: 10
        },
        events: {},
        zIndex: 6
    }, iT.prototype.nonDOMEvents = ["add", "afterUpdate", "drag", "remove"], q.compose(iT), function (t) {
        t.compose = function (t) {
            return t.navigation || (t.navigation = new i(t)), t
        };

        class i {
            constructor(t) {
                this.updates = [], this.chart = t
            }

            addUpdate(t) {
                this.chart.navigation.updates.push(t)
            }

            update(t, i) {
                this.updates.forEach(e => {
                    e.call(this.chart, t, i)
                })
            }
        }

        t.Additions = i
    }(r || (r = {}));
    let iN = r, {defined: iD, isNumber: iL, pick: iI} = u(), iS = {
        backgroundColor: "string",
        borderColor: "string",
        borderRadius: "string",
        color: "string",
        fill: "string",
        fontSize: "string",
        labels: "string",
        name: "string",
        stroke: "string",
        title: "string"
    }, iY = {
        annotationsFieldsTypes: iS, getAssignedAxis: function (t) {
            return t.filter(t => {
                let i = t.axis.getExtremes(), e = i.min, s = i.max, o = iI(t.axis.minPointOffset, 0);
                return iL(e) && iL(s) && t.value >= e - o && t.value <= s + o && !t.axis.options.isInternal
            })[0]
        }, getFieldType: function (t, i) {
            let e = iS[t], s = typeof i;
            return iD(e) && (s = e), ({string: "text", number: "number", boolean: "checkbox"})[s]
        }
    }, {getAssignedAxis: iX} = iY, {isNumber: iF, merge: iR} = u(), iU = {
        lang: {
            navigation: {
                popup: {
                    simpleShapes: "Simple shapes",
                    lines: "Lines",
                    circle: "Circle",
                    ellipse: "Ellipse",
                    rectangle: "Rectangle",
                    label: "Label",
                    shapeOptions: "Shape options",
                    typeOptions: "Details",
                    fill: "Fill",
                    format: "Text",
                    strokeWidth: "Line width",
                    stroke: "Line color",
                    title: "Title",
                    name: "Name",
                    labelOptions: "Label options",
                    labels: "Labels",
                    backgroundColor: "Background color",
                    backgroundColors: "Background colors",
                    borderColor: "Border color",
                    borderRadius: "Border radius",
                    borderWidth: "Border width",
                    style: "Style",
                    padding: "Padding",
                    fontSize: "Font size",
                    color: "Color",
                    height: "Height",
                    shapes: "Shape options"
                }
            }
        }, navigation: {
            bindingsClassName: "highcharts-bindings-container", bindings: {
                circleAnnotation: {
                    className: "highcharts-circle-annotation", start: function (t) {
                        let i = this.chart.pointer?.getCoordinates(t), e = i && iX(i.xAxis), s = i && iX(i.yAxis),
                            o = this.chart.options.navigation;
                        if (e && s) return this.chart.addAnnotation(iR({
                            langKey: "circle",
                            type: "basicAnnotation",
                            shapes: [{
                                type: "circle",
                                point: {x: e.value, y: s.value, xAxis: e.axis.index, yAxis: s.axis.index},
                                r: 5
                            }]
                        }, o.annotationsOptions, o.bindings.circleAnnotation.annotationsOptions))
                    }, steps: [function (t, i) {
                        let e;
                        let s = i.options.shapes, o = s && s[0] && s[0].point || {};
                        if (iF(o.xAxis) && iF(o.yAxis)) {
                            let i = this.chart.inverted, s = this.chart.xAxis[o.xAxis].toPixels(o.x),
                                n = this.chart.yAxis[o.yAxis].toPixels(o.y);
                            e = Math.max(Math.sqrt(Math.pow(i ? n - t.chartX : s - t.chartX, 2) + Math.pow(i ? s - t.chartY : n - t.chartY, 2)), 5)
                        }
                        i.update({shapes: [{r: e}]})
                    }]
                }, ellipseAnnotation: {
                    className: "highcharts-ellipse-annotation", start: function (t) {
                        let i = this.chart.pointer?.getCoordinates(t), e = i && iX(i.xAxis), s = i && iX(i.yAxis),
                            o = this.chart.options.navigation;
                        if (e && s) return this.chart.addAnnotation(iR({
                            langKey: "ellipse",
                            type: "basicAnnotation",
                            shapes: [{
                                type: "ellipse",
                                xAxis: e.axis.index,
                                yAxis: s.axis.index,
                                points: [{x: e.value, y: s.value}, {x: e.value, y: s.value}],
                                ry: 1
                            }]
                        }, o.annotationsOptions, o.bindings.ellipseAnnotation.annotationOptions))
                    }, steps: [function (t, i) {
                        let e = i.shapes[0], s = e.getAbsolutePosition(e.points[1]);
                        e.translatePoint(t.chartX - s.x, t.chartY - s.y, 1), e.redraw(!1)
                    }, function (t, i) {
                        let e = i.shapes[0], s = e.getAbsolutePosition(e.points[0]),
                            o = e.getAbsolutePosition(e.points[1]), n = e.getDistanceFromLine(s, o, t.chartX, t.chartY),
                            a = e.getYAxis(), r = Math.abs(a.toValue(0) - a.toValue(n));
                        e.setYRadius(r), e.redraw(!1)
                    }]
                }, rectangleAnnotation: {
                    className: "highcharts-rectangle-annotation", start: function (t) {
                        let i = this.chart.pointer?.getCoordinates(t), e = i && iX(i.xAxis), s = i && iX(i.yAxis);
                        if (!e || !s) return;
                        let o = e.value, n = s.value, a = e.axis.index, r = s.axis.index,
                            h = this.chart.options.navigation;
                        return this.chart.addAnnotation(iR({
                            langKey: "rectangle",
                            type: "basicAnnotation",
                            shapes: [{
                                type: "path",
                                points: [{xAxis: a, yAxis: r, x: o, y: n}, {xAxis: a, yAxis: r, x: o, y: n}, {
                                    xAxis: a,
                                    yAxis: r,
                                    x: o,
                                    y: n
                                }, {xAxis: a, yAxis: r, x: o, y: n}, {command: "Z"}]
                            }]
                        }, h.annotationsOptions, h.bindings.rectangleAnnotation.annotationsOptions))
                    }, steps: [function (t, i) {
                        let e = i.options.shapes, s = e && e[0] && e[0].points || [],
                            o = this.chart.pointer?.getCoordinates(t), n = o && iX(o.xAxis), a = o && iX(o.yAxis);
                        if (n && a) {
                            let t = n.value, e = a.value;
                            s[1].x = t, s[2].x = t, s[2].y = e, s[3].y = e, i.update({shapes: [{points: s}]})
                        }
                    }]
                }, labelAnnotation: {
                    className: "highcharts-label-annotation", start: function (t) {
                        let i = this.chart.pointer?.getCoordinates(t), e = i && iX(i.xAxis), s = i && iX(i.yAxis),
                            o = this.chart.options.navigation;
                        if (e && s) return this.chart.addAnnotation(iR({
                            langKey: "label",
                            type: "basicAnnotation",
                            labelOptions: {format: "{y:.2f}", overflow: "none", crop: !0},
                            labels: [{point: {xAxis: e.axis.index, yAxis: s.axis.index, x: e.value, y: s.value}}]
                        }, o.annotationsOptions, o.bindings.labelAnnotation.annotationsOptions))
                    }
                }
            }, events: {}, annotationsOptions: {animation: {defer: 0}}
        }
    }, {setOptions: iW} = u(), {format: iH} = tf(), {composed: iV, doc: ij, win: iq} = u(), {
        getAssignedAxis: iz,
        getFieldType: i_
    } = iY, {
        addEvent: iG,
        attr: iK,
        defined: iZ,
        fireEvent: i$,
        isArray: iJ,
        isFunction: iQ,
        isNumber: i0,
        isObject: i1,
        merge: i9,
        objectEach: i2,
        pick: i4,
        pushUnique: i5
    } = u();

    function i6() {
        this.chart.navigationBindings && this.chart.navigationBindings.deselectAnnotation()
    }

    function i7() {
        this.navigationBindings && this.navigationBindings.destroy()
    }

    function i3() {
        let t = this.options;
        t && t.navigation && t.navigation.bindings && (this.navigationBindings = new es(this, t.navigation), this.navigationBindings.initEvents(), this.navigationBindings.initUpdate())
    }

    function i8() {
        let t = this.navigationBindings, i = "highcharts-disabled-btn";
        if (this && t) {
            let e = !1;
            if (this.series.forEach(t => {
                !t.options.isInternal && t.visible && (e = !0)
            }), this.navigationBindings && this.navigationBindings.container && this.navigationBindings.container[0]) {
                let s = this.navigationBindings.container[0];
                i2(t.boundClassNames, (t, o) => {
                    let n = s.querySelectorAll("." + o);
                    if (n) for (let s = 0; s < n.length; s++) {
                        let o = n[s], a = o.className;
                        "normal" === t.noDataState ? -1 !== a.indexOf(i) && o.classList.remove(i) : e ? -1 !== a.indexOf(i) && o.classList.remove(i) : -1 === a.indexOf(i) && (o.className += " " + i)
                    }
                })
            }
        }
    }

    function et() {
        this.deselectAnnotation()
    }

    function ei() {
        this.selectedButtonElement = null
    }

    function ee(t) {
        let i, e;
        let s = t.prototype.defaultOptions.events && t.prototype.defaultOptions.events.click;

        function o(t) {
            let i = this, e = i.chart.navigationBindings, o = e.activeAnnotation;
            s && s.call(i, t), o !== i ? (e.deselectAnnotation(), e.activeAnnotation = i, i.setControlPointsVisibility(!0), i$(e, "showPopup", {
                annotation: i,
                formType: "annotation-toolbar",
                options: e.annotationToFields(i),
                onSubmit: function (t) {
                    if ("remove" === t.actionType) e.activeAnnotation = !1, e.chart.removeAnnotation(i); else {
                        let s = {};
                        e.fieldsToOptions(t.fields, s), e.deselectAnnotation();
                        let o = s.typeOptions;
                        "measure" === i.options.type && (o.crosshairY.enabled = 0 !== o.crosshairY.strokeWidth, o.crosshairX.enabled = 0 !== o.crosshairX.strokeWidth), i.update(s)
                    }
                }
            })) : i$(e, "closePopup"), t.activeAnnotation = !0
        }

        i9(!0, t.prototype.defaultOptions.events, {
            click: o, touchstart: function (t) {
                i = t.touches[0].clientX, e = t.touches[0].clientY
            }, touchend: function (t) {
                i && Math.sqrt(Math.pow(i - t.changedTouches[0].clientX, 2) + Math.pow(e - t.changedTouches[0].clientY, 2)) >= 4 || o.call(this, t)
            }
        })
    }

    class es {
        static compose(t, i) {
            i5(iV, "NavigationBindings") && (iG(t, "remove", i6), ee(t), i2(t.types, t => {
                ee(t)
            }), iG(i, "destroy", i7), iG(i, "load", i3), iG(i, "render", i8), iG(es, "closePopup", et), iG(es, "deselectButton", ei), iW(iU))
        }

        constructor(t, i) {
            this.boundClassNames = void 0, this.chart = t, this.options = i, this.eventsToUnbind = [], this.container = this.chart.container.getElementsByClassName(this.options.bindingsClassName || ""), this.container.length || (this.container = ij.getElementsByClassName(this.options.bindingsClassName || ""))
        }

        getCoords(t) {
            let i = this.chart.pointer?.getCoordinates(t);
            return [i && iz(i.xAxis), i && iz(i.yAxis)]
        }

        initEvents() {
            let t = this, i = t.chart, e = t.container, s = t.options;
            t.boundClassNames = {}, i2(s.bindings || {}, i => {
                t.boundClassNames[i.className] = i
            }), [].forEach.call(e, i => {
                t.eventsToUnbind.push(iG(i, "click", e => {
                    let s = t.getButtonEvents(i, e);
                    s && !s.button.classList.contains("highcharts-disabled-btn") && t.bindingsButtonClick(s.button, s.events, e)
                }))
            }), i2(s.events || {}, (i, e) => {
                iQ(i) && t.eventsToUnbind.push(iG(t, e, i, {passive: !1}))
            }), t.eventsToUnbind.push(iG(i.container, "click", function (e) {
                !i.cancelClick && i.isInsidePlot(e.chartX - i.plotLeft, e.chartY - i.plotTop, {visiblePlotOnly: !0}) && t.bindingsChartClick(this, e)
            })), t.eventsToUnbind.push(iG(i.container, u().isTouchDevice ? "touchmove" : "mousemove", function (i) {
                t.bindingsContainerMouseMove(this, i)
            }, u().isTouchDevice ? {passive: !1} : void 0))
        }

        initUpdate() {
            let t = this;
            iN.compose(this.chart).navigation.addUpdate(i => {
                t.update(i)
            })
        }

        bindingsButtonClick(t, i, e) {
            let s = this.chart, o = s.renderer.boxWrapper, n = !0;
            this.selectedButtonElement && (this.selectedButtonElement.classList === t.classList && (n = !1), i$(this, "deselectButton", {button: this.selectedButtonElement}), this.nextEvent && (this.currentUserDetails && "annotations" === this.currentUserDetails.coll && s.removeAnnotation(this.currentUserDetails), this.mouseMoveEvent = this.nextEvent = !1)), n ? (this.selectedButton = i, this.selectedButtonElement = t, i$(this, "selectButton", {button: t}), i.init && i.init.call(this, t, e), (i.start || i.steps) && s.renderer.boxWrapper.addClass("highcharts-draw-mode")) : (s.stockTools && t.classList.remove("highcharts-active"), o.removeClass("highcharts-draw-mode"), this.nextEvent = !1, this.mouseMoveEvent = !1, this.selectedButton = null)
        }

        bindingsChartClick(t, i) {
            t = this.chart;
            let e = this.activeAnnotation, s = this.selectedButton, o = t.renderer.boxWrapper;
            e && (e.cancelClick || i.activeAnnotation || !i.target.parentNode || function (t, i) {
                let e = iq.Element.prototype, s = e.matches || e.msMatchesSelector || e.webkitMatchesSelector, o = null;
                if (e.closest) o = e.closest.call(t, i); else do {
                    if (s.call(t, i)) return t;
                    t = t.parentElement || t.parentNode
                } while (null !== t && 1 === t.nodeType);
                return o
            }(i.target, ".highcharts-popup") ? e.cancelClick && setTimeout(() => {
                e.cancelClick = !1
            }, 0) : i$(this, "closePopup")), s && s.start && (this.nextEvent ? (this.nextEvent(i, this.currentUserDetails), this.steps && (this.stepIndex++, s.steps[this.stepIndex] ? this.mouseMoveEvent = this.nextEvent = s.steps[this.stepIndex] : (i$(this, "deselectButton", {button: this.selectedButtonElement}), o.removeClass("highcharts-draw-mode"), s.end && s.end.call(this, i, this.currentUserDetails), this.nextEvent = !1, this.mouseMoveEvent = !1, this.selectedButton = null))) : (this.currentUserDetails = s.start.call(this, i), this.currentUserDetails && s.steps ? (this.stepIndex = 0, this.steps = !0, this.mouseMoveEvent = this.nextEvent = s.steps[this.stepIndex]) : (i$(this, "deselectButton", {button: this.selectedButtonElement}), o.removeClass("highcharts-draw-mode"), this.steps = !1, this.selectedButton = null, s.end && s.end.call(this, i, this.currentUserDetails))))
        }

        bindingsContainerMouseMove(t, i) {
            this.mouseMoveEvent && this.mouseMoveEvent(i, this.currentUserDetails)
        }

        fieldsToOptions(t, i) {
            return i2(t, (t, e) => {
                let s = parseFloat(t), o = e.split("."), n = o.length - 1;
                if (!i0(s) || t.match(/px|em/g) || e.match(/format/g) || (t = s), "undefined" !== t) {
                    let e = i;
                    o.forEach((i, s) => {
                        if ("__proto__" !== i && "constructor" !== i) {
                            let a = i4(o[s + 1], "");
                            n === s ? e[i] = t : (e[i] || (e[i] = a.match(/\d/g) ? [] : {}), e = e[i])
                        }
                    })
                }
            }), i
        }

        deselectAnnotation() {
            this.activeAnnotation && (this.activeAnnotation.setControlPointsVisibility(!1), this.activeAnnotation = !1)
        }

        annotationToFields(t) {
            let i = t.options, e = es.annotationsEditable, s = e.nestedOptions,
                o = i4(i.type, i.shapes && i.shapes[0] && i.shapes[0].type, i.labels && i.labels[0] && i.labels[0].type, "label"),
                n = es.annotationsNonEditable[i.langKey] || [], a = {langKey: i.langKey, type: o};

            function r(i, e, o, a, h) {
                let l;
                o && iZ(i) && -1 === n.indexOf(e) && ((o.indexOf && o.indexOf(e)) >= 0 || o[e] || !0 === o) && (iJ(i) ? (a[e] = [], i.forEach((t, i) => {
                    i1(t) ? (a[e][i] = {}, i2(t, (t, o) => {
                        r(t, o, s[e], a[e][i], e)
                    })) : r(t, 0, s[e], a[e], e)
                })) : i1(i) ? (l = {}, iJ(a) ? (a.push(l), l[e] = {}, l = l[e]) : a[e] = l, i2(i, (t, i) => {
                    r(t, i, 0 === e ? o : s[e], l, e)
                })) : "format" === e ? a[e] = [iH(i, t.labels[0].points[0]).toString(), "text"] : iJ(a) ? a.push([i, i_(h, i)]) : a[e] = [i, i_(e, i)])
            }

            return i2(i, (t, n) => {
                "typeOptions" === n ? (a[n] = {}, i2(i[n], (t, i) => {
                    r(t, i, s, a[n], i)
                })) : r(t, n, e[o], a, n)
            }), a
        }

        getClickedClassNames(t, i) {
            let e = i.target, s = [], o;
            for (; e && e.tagName && ((o = iK(e, "class")) && (s = s.concat(o.split(" ").map(t => [t, e]))), (e = e.parentNode) !== t);) ;
            return s
        }

        getButtonEvents(t, i) {
            let e;
            let s = this;
            return this.getClickedClassNames(t, i).forEach(t => {
                s.boundClassNames[t[0]] && !e && (e = {events: s.boundClassNames[t[0]], button: t[1]})
            }), e
        }

        update(t) {
            this.options = i9(!0, this.options, t), this.removeEvents(), this.initEvents()
        }

        removeEvents() {
            this.eventsToUnbind.forEach(t => t())
        }

        destroy() {
            this.removeEvents()
        }
    }

    es.annotationsEditable = {
        nestedOptions: {
            labelOptions: ["style", "format", "backgroundColor"],
            labels: ["style"],
            label: ["style"],
            style: ["fontSize", "color"],
            background: ["fill", "strokeWidth", "stroke"],
            innerBackground: ["fill", "strokeWidth", "stroke"],
            outerBackground: ["fill", "strokeWidth", "stroke"],
            shapeOptions: ["fill", "strokeWidth", "stroke"],
            shapes: ["fill", "strokeWidth", "stroke"],
            line: ["strokeWidth", "stroke"],
            backgroundColors: [!0],
            connector: ["fill", "strokeWidth", "stroke"],
            crosshairX: ["strokeWidth", "stroke"],
            crosshairY: ["strokeWidth", "stroke"]
        },
        circle: ["shapes"],
        ellipse: ["shapes"],
        verticalLine: [],
        label: ["labelOptions"],
        measure: ["background", "crosshairY", "crosshairX"],
        fibonacci: [],
        tunnel: ["background", "line", "height"],
        pitchfork: ["innerBackground", "outerBackground"],
        rect: ["shapes"],
        crookedLine: [],
        basicAnnotation: ["shapes", "labelOptions"]
    }, es.annotationsNonEditable = {
        rectangle: ["crosshairX", "crosshairY", "labelOptions"],
        ellipse: ["labelOptions"],
        circle: ["labelOptions"]
    };
    let eo = u();
    eo.Annotation = eo.Annotation || iT, eo.NavigationBindings = eo.NavigationBindings || es, eo.Annotation.compose(eo.Chart, eo.NavigationBindings, eo.Pointer, eo.SVGRenderer);
    let en = u();
    return p.default
})());