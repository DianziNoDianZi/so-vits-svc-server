// 通用上传进度：拦截 .js-upload 表单，XHR 上传并显示进度，成功后跳转
(function () {
    "use strict";

    function fmtBytes(b) {
        if (b >= 1073741824) return (b / 1073741824).toFixed(2) + " GB";
        if (b >= 1048576) return (b / 1048576).toFixed(1) + " MB";
        return (b / 1024).toFixed(0) + " KB";
    }

    document.addEventListener("DOMContentLoaded", function () {
        var forms = document.querySelectorAll("form.js-upload");
        if (!forms.length) return;

        forms.forEach(function (form) {
            form.addEventListener("submit", function (e) {
                e.preventDefault();
                if (form.dataset.uploading === "1") return;
                form.dataset.uploading = "1";

                var overlay = document.getElementById("uploadOverlay");
                if (!overlay) { form.submit(); return; }
                var bar = document.getElementById("upBar");
                var pct = document.getElementById("upPct");
                var meta = document.getElementById("upMeta");
                var msg = document.getElementById("upMsg");

                overlay.style.display = "flex";
                if (bar) bar.style.width = "0%";
                if (pct) pct.textContent = "0%";
                if (meta) meta.textContent = "正在连接...";
                if (msg) msg.textContent = "正在上传，请稍候";

                var fd = new FormData(form);
                var xhr = new XMLHttpRequest();
                var start = Date.now();

                xhr.open("POST", form.action, true);

                xhr.upload.onprogress = function (ev) {
                    if (!meta) return;
                    if (!ev.lengthComputable) {
                        meta.textContent = "已上传 " + fmtBytes(ev.loaded);
                        return;
                    }
                    var p = Math.round((ev.loaded * 100) / ev.total);
                    if (bar) bar.style.width = p + "%";
                    if (pct) pct.textContent = p + "%";
                    var secs = (Date.now() - start) / 1000;
                    var speed = secs > 0 ? ev.loaded / secs : 0;
                    meta.textContent = "已上传 " + fmtBytes(ev.loaded) + " / " + fmtBytes(ev.total) +
                        " · " + fmtBytes(speed) + "/s";
                };

                xhr.onload = function () {
                    if (xhr.status >= 200 && xhr.status < 400) {
                        if (msg) msg.textContent = "上传完成，跳转中...";
                        var url = xhr.responseURL;
                        if (url && url !== location.href) {
                            window.location.href = url;
                        } else {
                            window.location.reload();
                        }
                    } else {
                        overlay.style.display = "none";
                        form.dataset.uploading = "";
                        if (msg) msg.textContent = "上传失败（" + xhr.status + "）";
                        try {
                            var data = JSON.parse(xhr.responseText);
                            alert(data.msg || "上传失败：" + xhr.status);
                        } catch (err) {
                            alert("上传失败：" + xhr.status);
                        }
                    }
                };

                xhr.onerror = function () {
                    overlay.style.display = "none";
                    form.dataset.uploading = "";
                    if (msg) msg.textContent = "网络错误";
                    alert("上传失败：网络错误，请重试");
                };

                xhr.send(fd);
            });
        });
    });

    // 通用"处理中"覆盖层：非上传表单提交，超过 300ms 未返回则显示转圈+耗时
    // （正常表单提交会整页跳转，覆盖层随页面卸载自然消失；快速操作不会闪）
    (function () {
        document.addEventListener("DOMContentLoaded", function () {
            var overlay = document.getElementById("processingOverlay");
            if (!overlay) return;
            var msgEl = document.getElementById("procMsg");
            var elapsedEl = document.getElementById("procElapsed");
            var forms = document.querySelectorAll("form:not(.js-upload)");
            forms.forEach(function (form) {
                form.addEventListener("submit", function () {
                    setTimeout(function () {
                        overlay.style.display = "flex";
                        var start = Date.now();
                        if (msgEl) msgEl.textContent = "正在处理，请稍候";
                        setInterval(function () {
                            var s = Math.floor((Date.now() - start) / 1000);
                            if (elapsedEl) elapsedEl.textContent = s + "s";
                        }, 500);
                    }, 300);
                });
            });
        });
    })();
})();
