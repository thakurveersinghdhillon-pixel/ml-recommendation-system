"use client";

import React, { useState } from "react";
import Link from "next/link";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import {
  Users,
  UserPlus,
  Search,
  UserCheck,
  Trash2,
  Edit2,
  AlertCircle,
  ArrowRight,
  ShieldAlert,
} from "lucide-react";
import { createUser, getUser, updateUser, deleteUser } from "@/lib/api/users";
import { User } from "@/types/user";
import { useUser } from "@/context/UserContext";
import { useToast } from "@/context/ToastContext";
import { ConfirmationModal } from "@/components/ui/ConfirmationModal";

const createUserSchema = z.object({
  username: z.string().min(3, "Username must be at least 3 characters").max(50, "Username too long"),
  email: z.string().email("Invalid email address"),
});

type CreateUserFormData = z.infer<typeof createUserSchema>;

export default function UsersPage() {
  const { activeUser, setActiveUser } = useUser();
  const { showToast } = useToast();

  const [lookupId, setLookupId] = useState<string>("");
  const [selectedUser, setSelectedUser] = useState<User | null>(activeUser);
  const [isSearching, setIsSearching] = useState<boolean>(false);
  const [searchError, setSearchError] = useState<string | null>(null);

  // Edit Mode state
  const [isEditing, setIsEditing] = useState(false);
  const [editUsername, setEditUsername] = useState("");
  const [editEmail, setEditEmail] = useState("");

  // Delete Modal state
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  const {
    register,
    handleSubmit,
    reset,
    formState: { errors, isSubmitting },
  } = useForm<CreateUserFormData>({
    resolver: zodResolver(createUserSchema),
  });

  const handleCreateUser = async (data: CreateUserFormData) => {
    try {
      const newUser = await createUser(data);
      showToast(`User '${newUser.username}' created successfully!`, "success");
      setSelectedUser(newUser);
      setActiveUser(newUser);
      reset();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to create user.";
      showToast(msg, "error");
    }
  };

  const handleLookupUser = async (e: React.FormEvent) => {
    e.preventDefault();
    const idNum = Number(lookupId);
    if (!idNum || idNum <= 0) {
      setSearchError("Please enter a valid positive integer User ID.");
      return;
    }

    try {
      setIsSearching(true);
      setSearchError(null);
      const user = await getUser(idNum);
      setSelectedUser(user);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "User not found.";
      setSearchError(msg);
      setSelectedUser(null);
    } finally {
      setIsSearching(false);
    }
  };

  const handleStartEdit = () => {
    if (!selectedUser) return;
    setEditUsername(selectedUser.username);
    setEditEmail(selectedUser.email);
    setIsEditing(true);
  };

  const handleSaveEdit = async () => {
    if (!selectedUser) return;
    try {
      const updated = await updateUser(selectedUser.id, {
        username: editUsername,
        email: editEmail,
      });
      setSelectedUser(updated);
      if (activeUser?.id === updated.id) {
        setActiveUser(updated);
      }
      showToast("User profile updated successfully!", "success");
      setIsEditing(false);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to update user.";
      showToast(msg, "error");
    }
  };

  const handleDeleteConfirm = async () => {
    if (!selectedUser) return;
    try {
      setIsDeleting(true);
      await deleteUser(selectedUser.id);
      showToast(`User #${selectedUser.id} deleted successfully.`, "success");
      if (activeUser?.id === selectedUser.id) {
        setActiveUser(null);
      }
      setSelectedUser(null);
      setDeleteModalOpen(false);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to delete user.";
      showToast(msg, "error");
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      {/* Header */}
      <div className="border-b border-slate-800 pb-6 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-black text-slate-100 flex items-center gap-3">
            <Users className="w-8 h-8 text-amber-400" />
            User Account Management
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Register new users, lookup profiles, manage ratings, and switch active system profiles
          </p>
        </div>

        {activeUser && (
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-300 text-xs font-medium">
            <UserCheck className="w-4 h-4 text-emerald-400" />
            <span>Active: {activeUser.username} (#{activeUser.id})</span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Left Column: Create User Form */}
        <div className="p-6 rounded-3xl bg-slate-900/90 border border-slate-800 space-y-6 shadow-xl">
          <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
            <UserPlus className="w-5 h-5 text-amber-400" />
            <h2 className="text-base font-bold text-slate-100">Create New User</h2>
          </div>

          <form onSubmit={handleSubmit(handleCreateUser)} className="space-y-4">
            <div className="space-y-1">
              <label className="text-xs font-semibold text-slate-300">Username</label>
              <input
                {...register("username")}
                type="text"
                placeholder="e.g. cinema_lover"
                className="w-full bg-slate-950 border border-slate-800 focus:border-amber-500 text-slate-100 px-3.5 py-2.5 rounded-xl text-xs font-medium focus:outline-none"
              />
              {errors.username && (
                <p className="text-[11px] text-rose-400">{errors.username.message}</p>
              )}
            </div>

            <div className="space-y-1">
              <label className="text-xs font-semibold text-slate-300">Email Address</label>
              <input
                {...register("email")}
                type="email"
                placeholder="e.g. lover@example.com"
                className="w-full bg-slate-950 border border-slate-800 focus:border-amber-500 text-slate-100 px-3.5 py-2.5 rounded-xl text-xs font-medium focus:outline-none"
              />
              {errors.email && (
                <p className="text-[11px] text-rose-400">{errors.email.message}</p>
              )}
            </div>

            <button
              type="submit"
              disabled={isSubmitting}
              className="w-full bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold py-2.5 rounded-xl text-xs transition-all shadow-lg shadow-amber-500/20 flex items-center justify-center gap-2 disabled:opacity-50"
            >
              <UserPlus className="w-4 h-4" />
              <span>{isSubmitting ? "Creating User..." : "Register User Account"}</span>
            </button>
          </form>
        </div>

        {/* Right Column: User Lookup & Selected User Details */}
        <div className="space-y-6">
          {/* Lookup Form */}
          <div className="p-6 rounded-3xl bg-slate-900/90 border border-slate-800 space-y-4 shadow-xl">
            <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
              <Search className="w-5 h-5 text-amber-400" />
              <h2 className="text-base font-bold text-slate-100">Lookup User Profile</h2>
            </div>

            <form onSubmit={handleLookupUser} className="flex gap-2">
              <input
                type="number"
                value={lookupId}
                onChange={(e) => setLookupId(e.target.value)}
                placeholder="Enter User ID (e.g. 1)"
                className="flex-1 bg-slate-950 border border-slate-800 focus:border-amber-500 text-slate-100 px-3.5 py-2 rounded-xl text-xs font-medium focus:outline-none"
              />
              <button
                type="submit"
                disabled={isSearching}
                className="bg-slate-800 hover:bg-slate-700 text-slate-200 font-bold px-4 py-2 rounded-xl text-xs transition-all border border-slate-700"
              >
                Lookup
              </button>
            </form>

            {searchError && (
              <p className="text-xs text-rose-400 flex items-center gap-1.5">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{searchError}</span>
              </p>
            )}
          </div>

          {/* Selected User Details Card */}
          {selectedUser && (
            <div className="p-6 rounded-3xl bg-slate-900 border border-amber-500/30 space-y-5 shadow-2xl relative overflow-hidden">
              <div className="flex items-start justify-between border-b border-slate-800 pb-4">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-lg text-slate-100">
                      {selectedUser.username}
                    </span>
                    <span className="text-xs text-amber-400 font-mono bg-amber-500/10 border border-amber-500/20 px-2 py-0.5 rounded-full">
                      ID #{selectedUser.id}
                    </span>
                  </div>
                  <p className="text-xs text-slate-400">{selectedUser.email}</p>
                </div>

                {activeUser?.id === selectedUser.id ? (
                  <span className="text-[10px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 px-2.5 py-1 rounded-full">
                    Active User
                  </span>
                ) : (
                  <button
                    onClick={() => {
                      setActiveUser(selectedUser);
                      showToast(`Set '${selectedUser.username}' as active profile!`, "success");
                    }}
                    className="text-xs font-semibold bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 border border-amber-500/30 px-3 py-1 rounded-xl transition-all"
                  >
                    Set Active
                  </button>
                )}
              </div>

              {/* Edit Mode Inline Controls */}
              {isEditing ? (
                <div className="space-y-3 bg-slate-950 p-4 rounded-2xl border border-slate-800">
                  <h4 className="text-xs font-bold text-slate-200">Edit User Details</h4>
                  <input
                    type="text"
                    value={editUsername}
                    onChange={(e) => setEditUsername(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-800 text-slate-100 text-xs px-3 py-1.5 rounded-lg"
                    placeholder="Username"
                  />
                  <input
                    type="email"
                    value={editEmail}
                    onChange={(e) => setEditEmail(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-800 text-slate-100 text-xs px-3 py-1.5 rounded-lg"
                    placeholder="Email"
                  />
                  <div className="flex gap-2 justify-end">
                    <button
                      onClick={() => setIsEditing(false)}
                      className="text-xs px-3 py-1 rounded-lg text-slate-400 hover:text-slate-200"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleSaveEdit}
                      className="text-xs bg-amber-500 text-slate-950 font-bold px-3 py-1 rounded-lg"
                    >
                      Save Changes
                    </button>
                  </div>
                </div>
              ) : null}

              {/* Action Buttons */}
              <div className="flex items-center justify-between pt-2">
                <Link
                  href={`/users/${selectedUser.id}`}
                  className="inline-flex items-center gap-2 text-xs font-bold text-amber-400 hover:text-amber-300"
                >
                  <span>View Full Profile & Ratings</span>
                  <ArrowRight className="w-4 h-4" />
                </Link>

                <div className="flex items-center gap-2">
                  <button
                    onClick={handleStartEdit}
                    className="p-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
                    title="Edit User"
                  >
                    <Edit2 className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => setDeleteModalOpen(true)}
                    className="p-2 rounded-xl bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border border-rose-500/20 transition-colors"
                    title="Delete User"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Confirmation Modal */}
      <ConfirmationModal
        isOpen={deleteModalOpen}
        title="Delete User Profile"
        message={`Are you sure you want to delete user '${selectedUser?.username}' (#${selectedUser?.id})? This will cascade and delete all associated ratings and recommendation history.`}
        confirmLabel="Delete User"
        onConfirm={handleDeleteConfirm}
        onCancel={() => setDeleteModalOpen(false)}
        isConfirming={isDeleting}
      />
    </div>
  );
}
