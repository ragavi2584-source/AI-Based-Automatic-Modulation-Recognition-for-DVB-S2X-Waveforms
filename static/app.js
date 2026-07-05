/*
==========================================
AI-Based Automatic Modulation Recognition
app.js
==========================================
*/

document.addEventListener("DOMContentLoaded", function () {

    const uploadForm = document.querySelector("form");

    const fileInput = document.querySelector(
        "input[type='file']"
    );

    const submitButton = document.querySelector(
        "button[type='submit']"
    );

    // -----------------------------
    // Validate File
    // -----------------------------
    if (fileInput) {

        fileInput.addEventListener(
            "change",
            function () {

                const file = this.files[0];

                if (!file)
                    return;

                if (!file.name.endsWith(".npy")) {

                    alert(
                        "Only .npy files are allowed."
                    );

                    this.value = "";

                    return;
                }

                console.log(
                    "Selected File:",
                    file.name
                );

            }
        );
    }

    // -----------------------------
    // Upload Animation
    // -----------------------------
    if (uploadForm) {

        uploadForm.addEventListener(
            "submit",
            function () {

                submitButton.disabled = true;

                submitButton.innerHTML =
                    "Processing...";

            }
        );

    }

});
/*
==========================================
Extra UI Features
==========================================
*/

// -----------------------------
// Show Notification
// -----------------------------
function showNotification(message, type = "info") {

    const notification = document.createElement("div");

    notification.className = "flash";

    notification.innerHTML = message;

    document.body.prepend(notification);

    setTimeout(() => {
        notification.remove();
    }, 3000);
}

// -----------------------------
// Drag and Drop Upload
// -----------------------------
const uploadArea = document.querySelector(".upload-card");

if (uploadArea) {

    uploadArea.addEventListener("dragover", function (e) {

        e.preventDefault();

        uploadArea.style.border = "2px dashed #1565c0";

    });

    uploadArea.addEventListener("dragleave", function () {

        uploadArea.style.border = "";

    });

    uploadArea.addEventListener("drop", function (e) {

        e.preventDefault();

        uploadArea.style.border = "";

        const files = e.dataTransfer.files;

        if (files.length > 0) {

            const input = document.querySelector(
                "input[type='file']"
            );

            input.files = files;

            showNotification(
                "File selected successfully!",
                "success"
            );
        }

    });

}

// -----------------------------
// Reset Button (Optional)
// -----------------------------
const resetButton = document.getElementById("resetBtn");

if (resetButton) {

    resetButton.addEventListener("click", function () {

        const input = document.querySelector(
            "input[type='file']"
        );

        if (input) {

            input.value = "";

            showNotification(
                "Selection cleared."
            );

        }

    });

}

// -----------------------------
// Console Message
// -----------------------------
console.log(
    "AI-Based Automatic Modulation Recognition UI Loaded"
);